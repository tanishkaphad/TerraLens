"""Constrained natural-language to query-plan parsing."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PlanError(ValueError):
    """The request is unsupported or cannot be safely interpreted."""


class AttributeFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    field: str
    operator: str
    value: str


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concept_id: str
    visual_query: str
    reference_layer: str
    predicate: str
    distance_m: float | None
    attributes: list[AttributeFilter] = Field(default_factory=list)


def _normal(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _options(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    concepts = manifest.get("supported_concepts", {})
    layers = manifest.get("vector_layers", {})
    return concepts, layers


def _validate_plan(raw: dict[str, Any], manifest: dict[str, Any]) -> QueryPlan:
    concepts, layers = _options(manifest)
    try:
        plan = QueryPlan.model_validate(raw)
    except ValidationError as exc:
        raise PlanError(f"Could not interpret that request: {exc.errors()[0]['msg']}") from exc
    if plan.concept_id not in concepts:
        raise PlanError(f"Visual concept '{plan.concept_id}' is not indexed.")
    if not plan.visual_query.strip():
        raise PlanError("Clarify the visual concept before searching.")
    if plan.reference_layer not in layers:
        raise PlanError(f"Reference layer '{plan.reference_layer}' is not available.")
    if plan.predicate not in {"within_distance", "intersects"}:
        raise PlanError("Supported spatial relationships are 'within_distance' and 'intersects'.")
    if plan.predicate == "within_distance":
        if plan.distance_m is None or not 0 < plan.distance_m <= 5000:
            raise PlanError("Choose a distance from 1 to 5,000 metres.")
    elif plan.distance_m is not None:
        raise PlanError("An intersection query does not take a distance.")
    if len(plan.visual_query) > 160:
        raise PlanError("The visual search phrase is too long.")
    available_fields = set(manifest.get("queryable_attributes", []))
    for condition in plan.attributes:
        if condition.role not in {"subject", "reference"}:
            raise PlanError("Attribute role must be 'subject' or 'reference'.")
        if condition.field not in available_fields:
            raise PlanError(f"Attribute '{condition.field}' is not available in this data bundle.")
        if condition.operator not in {"equals", "contains"}:
            raise PlanError("Supported attribute checks are 'equals' and 'contains'.")
    return plan


def _rule_plan(query: str, manifest: dict[str, Any], previous: dict[str, Any] | None) -> QueryPlan:
    concepts, layers = _options(manifest)
    text = _normal(query)

    # A short follow-up only edits the radius on the prior, already validated plan.
    radius = re.search(r"\b(\d+(?:\.\d+)?)\s*(km|kilometres?|kilometers?|m|metres?|meters?)\b", text)
    if previous and radius and re.search(r"\b(make|change|set|increase|decrease|instead|radius)\b", text):
        amount = float(radius.group(1))
        unit = radius.group(2)
        new_distance = amount * 1000 if unit.startswith("k") else amount
        edited = dict(previous)
        edited["distance_m"] = new_distance
        return _validate_plan(edited, manifest)

    if any(word in text.split() for word in ("commercial", "industrial", "residential", "public")):
        raise PlanError("This query includes a land-use or building-use condition. Add a supported mapped attribute before using that condition.")
    if re.search(r"\b(without|lacking|no)\b", text):
        raise PlanError("A visual non-detection cannot prove an object is absent; negative visual queries need a separate validation policy.")

    matches: list[tuple[int, str, str]] = []
    for concept_id, details in concepts.items():
        labels = [concept_id.replace("_", " "), details.get("prompt", "")]
        labels.extend(details.get("aliases", []))
        for label in labels:
            normalized = _normal(str(label))
            if normalized and re.search(rf"\b{re.escape(normalized)}\b", text):
                matches.append((len(normalized), concept_id, str(label)))
    if not matches:
        supported = ", ".join(sorted(concepts)) or "none"
        raise PlanError(f"No indexed visual concept matched. Supported concepts: {supported}.")
    matches.sort(reverse=True)
    longest = [match for match in matches if match[0] == matches[0][0]]
    if len({match[1] for match in longest}) != 1:
        raise PlanError("The visual concept is ambiguous; specify one of the supported concepts.")
    _, concept_id, visual_query = matches[0]

    layer_matches: list[tuple[int, str]] = []
    for layer_id, details in layers.items():
        aliases = [layer_id.replace("_", " ")]
        if isinstance(details, dict):
            aliases.extend(details.get("aliases", []))
        for alias in aliases:
            normalized = _normal(str(alias))
            if normalized and re.search(rf"\b{re.escape(normalized)}\b", text):
                layer_matches.append((len(normalized), layer_id))
    if not layer_matches:
        raise PlanError("Name an available reference layer, such as roads or waterways.")
    layer_matches.sort(reverse=True)
    reference_layer = layer_matches[0][1]

    distance = re.search(r"\b(\d+(?:\.\d+)?)\s*(km|kilometres?|kilometers?|m|metres?|meters?)\b", text)
    if re.search(r"\b(intersect|intersecting|intersects|crossing)\b", text):
        predicate, distance_m = "intersects", None
    else:
        if not distance:
            raise PlanError("Specify a distance, for example 'within 200 metres of roads'.")
        amount, unit = float(distance.group(1)), distance.group(2)
        predicate, distance_m = "within_distance", amount * (1000 if unit.startswith("k") else 1)

    return _validate_plan(
        {
            "concept_id": concept_id,
            "visual_query": visual_query,
            "reference_layer": reference_layer,
            "predicate": predicate,
            "distance_m": distance_m,
            "attributes": [],
        },
        manifest,
    )


def _llm_plan(query: str, manifest: dict[str, Any], previous: dict[str, Any] | None) -> QueryPlan:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise PlanError("Install the 'openai' dependency to use the configured language parser.") from exc

    concepts, layers = _options(manifest)
    if not concepts or not layers:
        raise PlanError("This data bundle has no supported concepts or vector layers.")
    fields = sorted(set(manifest.get("queryable_attributes", [])))
    schema = {
        "type": "object",
        "properties": {
            "concept_id": {"type": "string", "enum": sorted(concepts)},
            "visual_query": {"type": "string"},
            "reference_layer": {"type": "string", "enum": sorted(layers)},
            "predicate": {"type": "string", "enum": ["within_distance", "intersects"]},
            "distance_m": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["subject", "reference"]},
                        "field": {"type": "string", "enum": fields or [""]},
                        "operator": {"type": "string", "enum": ["equals", "contains"]},
                        "value": {"type": "string"},
                    },
                    "required": ["role", "field", "operator", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["concept_id", "visual_query", "reference_layer", "predicate", "distance_m", "attributes"],
        "additionalProperties": False,
    }
    context = {
        "supported_concepts": concepts,
        "vector_layers": layers,
        "queryable_attributes": fields,
        "previous_validated_plan": previous,
    }
    response = OpenAI().chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate the user request into a plan using only the supplied concepts, layers and fields. "
                    "Never weaken or omit a requested condition. If information is missing, use an empty visual_query "
                    "or a null distance so validation asks the user to clarify. For a follow-up, preserve every prior "
                    "field the user did not request changing. Distances are metres."
                ),
            },
            {"role": "user", "content": json.dumps({"request": query, "capabilities": context})},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "geo_query_plan", "strict": True, "schema": schema}},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise PlanError("The language parser returned no query plan.")
    return _validate_plan(json.loads(content), manifest)


def plan_query(query: str, manifest: dict[str, Any], previous: dict[str, Any] | None = None) -> QueryPlan:
    if not query.strip() or len(query) > 500:
        raise PlanError("Enter a search up to 500 characters long.")
    if previous:
        radius = re.search(r"\b(\d+(?:\.\d+)?)\s*(km|kilometres?|kilometers?|m|metres?|meters?)\b", _normal(query))
        if radius and re.search(r"\b(make|change|set|increase|decrease|instead|radius)\b", _normal(query)):
            return _rule_plan(query, manifest, previous)
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"):
        return _llm_plan(query, manifest, previous)
    return _rule_plan(query, manifest, previous)
