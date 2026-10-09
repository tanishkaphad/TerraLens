"""Select registered evidence tools without a paid conversation provider."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .planner import PlanError, QueryPlan, _reject_unsupported_claims, _rule_plan, validate_plan


_IMAGERY = re.compile(r"^\s*(?:(?:find|show|search|locate)\s+)?(?:satellite\s+|aerial\s+)?(?:images|imagery|image\s+tiles)\s+(?:of|showing|with)\s+(.+)$", re.I)
_UNPLANNED_SPATIAL = re.compile(
    r"\b(?:(?:north|south|east|west|upstream|downstream)\s+of|"
    r"(?:nearest|closest|farthest)\s+to|across\s+from|next\s+to|"
    r"along|around|surrounded\s+by|between|inside|outside|adjacent|beside|overlapping|overlaps)\b",
    re.I,
)
_RELATION_START = re.compile(
    r"\b(?:within|intersect|intersects|intersecting|crossing|near|close\s+to)\b",
    re.I,
)


@dataclass(frozen=True)
class _EvidencePolicy:
    """A nonvisual claim and the structured fields that could establish it.

    This is a conservative claim guard, not a general language-understanding
    guarantee. New evidence domains can be added without changing dispatch.
    """

    name: str
    pattern: re.Pattern[str]
    fields: frozenset[str]
    evidence: str


def _policy(name: str, pattern: str, fields: Iterable[str], evidence: str) -> _EvidencePolicy:
    return _EvidencePolicy(name, re.compile(pattern, re.I), frozenset(fields), evidence)


_EVIDENCE_POLICIES = (
    _policy(
        "land use",
        r"\b(?:commercial|residential|industrial|retail|mixed[- ]use)\b",
        {"building", "building:use", "landuse", "use", "zoning", "highway", "amenity"},
        "a registered land-use, road-class, or zoning attribute",
    ),
    _policy(
        "ownership or access",
        r"\b(?:(?:government|municipal|state|publicly|privately)[- ]owned|owned\s+by|ownership|private|public)\b",
        {"owner", "ownership", "operator", "tenure", "access", "building", "building:use", "amenity"},
        "an attributable ownership, access, or facility record",
    ),
    _policy(
        "administrative status",
        r"\b(?:permits?|permitted|licensed|licen[cs]ed|registered|insured|certified|approved|code[- ]compliant)\b",
        {"permit", "permit_status", "license", "licence", "registration", "insurance", "insured", "certification", "compliance"},
        "an attributable permit, registry, insurance, or certification field",
    ),
    _policy(
        "market or financial status",
        r"\b(?:for\s+sale|for\s+rent|sale\s+price|property\s+value|market\s+value|affordable|profit(?:able)?|revenue|repair\s+budget)\b",
        {"sale_status", "rental_status", "price", "value", "valuation", "rent", "affordability", "income", "revenue", "cost", "budget"},
        "a dated market, valuation, income, or budget record",
    ),
    _policy(
        "occupancy or demographics",
        r"\b(?:vacant|vacancy|occupied|occupancy|residents?|population|households?|overcrowded|low[- ]income|high[- ]income|poverty|wealthy)\b",
        {"occupancy", "vacancy", "residents", "population", "households", "capacity", "income", "demographic"},
        "a dated occupancy, census, or demographic field",
    ),
    _policy(
        "live operational status",
        r"\b(?:(?:open|closed)\s+(?:now|today)|currently\s+(?:open|closed)|available\s+beds?|bed\s+availability|wait(?:ing)?\s+time|operational\s+status)\b",
        {"opening_hours", "status", "operational_status", "availability", "available_beds", "beds", "capacity", "last_updated"},
        "a current operational record with a timestamp",
    ),
    _policy(
        "construction or lifecycle status",
        r"\b(?:construction\s+sites?|under\s+construction|active\s+construction|new\s+construction|recent\s+construction|"
        r"under\s+renovation|unfinished|completed|renovated|demolished|abandoned|"
        r"new\s+(?:buildings?|homes?|houses?)|old\s+(?:buildings?|homes?|houses?)|"
        r"(?:buildings?|homes?|houses?)\s+(?:built|constructed)\b)",
        {"construction", "construction_status", "start_date", "end_date", "year_built", "building:condition", "lifecycle", "status"},
        "a dated construction, inspection, or lifecycle field",
    ),
    _policy(
        "structural or safety status",
        r"\b(?:structurally\s+(?:safe|unsafe|damaged|unsound)|at\s+risk\s+of\s+collapse|collapse\s+risk|"
        r"(?:damaged|collapsed)\s+(?:buildings?|structures?|homes?|houses?)|"
        r"condemned|earthquake[- ]resistant|fire\s+risk|safe\s+(?:buildings?|neighbou?rhoods?))\b",
        {"structural_status", "condition", "damage", "inspection_status", "collapse_risk", "fire_risk", "safety_status"},
        "a qualified inspection, engineering, hazard, or safety field",
    ),
    _policy(
        "crime or social status",
        r"\b(?:high[- ]crime|low[- ]crime|crime\s+rate|safe\s+area|unsafe\s+area|deprived|disadvantaged)\b",
        {"crime", "crime_rate", "safety_index", "deprivation", "social_index"},
        "a dated official crime or social-statistics field",
    ),
    _policy(
        "energy performance",
        r"\b(?:energy[- ]efficient|energy\s+rating|energy\s+performance|poorly\s+insulated|well[- ]insulated)\b",
        {"energy_rating", "energy_efficiency", "energy_performance", "epc", "insulation"},
        "an energy audit or building-performance field",
    ),
    _policy(
        "scientific measurement",
        r"\b(?:ndvi|nir|near[- ]infrared|biomass|carbon|soil\s+moisture|pollution|air\s+quality|water\s+quality|"
        r"surface\s+temperature|biodiversity|tree\s+species|crop\s+yield|solar\s+potential|suitable\s+for\s+solar|"
        r"(?:healthy|stressed|diseased)\s+(?:trees?|vegetation|crops?))\b",
        {"ndvi", "nir", "biomass", "carbon", "soil_moisture", "pollution", "air_quality", "water_quality", "temperature", "biodiversity", "species", "crop_yield", "crop_health", "solar_potential"},
        "a registered spectral, environmental, ecological, or field-measurement attribute",
    ),
    _policy(
        "hazard, terrain, or routing result",
        r"\b(?:flood[- ]prone|flood\s+risk|elevation|slope|travel\s+time|shortest\s+route|evacuation\s+route)\b",
        {"flood_risk", "hazard", "elevation", "slope", "travel_time", "route"},
        "a registered hazard, elevation, terrain, or routing result",
    ),
    _policy(
        "date or age",
        r"\b(?:(?:new|old|recent)\s+(?:roads?|roofs?|structures?|developments?|fields?|clearings?)|"
        r"(?:from|during|in)\s+(?:19|20)\d{2}|last\s+(?:year|month|week)|as\s+of\s+(?:today|yesterday))\b",
        {"acquisition_date", "date", "year", "year_built", "start_date", "end_date", "last_updated"},
        "a dated source or lifecycle attribute tied to the requested feature",
    ),
)


def _normal_tokens(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", value.casefold())
    return {word[:-1] if len(word) > 4 and word.endswith("s") else word for word in words}


def _plan_attributes(plan: QueryPlan):
    yield from plan.attributes
    for relation in plan.relations:
        yield from relation.attributes


def _condition_role(query: str, match: re.Match[str]) -> str:
    relation = _RELATION_START.search(query)
    return "reference" if relation and match.start() >= relation.start() else "subject"


def _condition_preserves(query: str, match: re.Match[str], policy: _EvidencePolicy, plan: QueryPlan) -> bool:
    requested = _normal_tokens(match.group(0))
    role = _condition_role(query, match)
    for condition in _plan_attributes(plan):
        if condition.role != role or condition.field.casefold() not in policy.fields:
            continue
        recorded = _normal_tokens(condition.field) | _normal_tokens(condition.value)
        if requested & recorded:
            return True
    return False


def _relevant_layers(query: str, match: re.Match[str], plan: QueryPlan, manifest: dict) -> set[str]:
    if _condition_role(query, match) == "reference":
        return {
            layer for layer in [plan.reference_layer, *(relation.reference_layer for relation in plan.relations)]
            if layer
        }
    if plan.subject_layer:
        return {plan.subject_layer}
    associated = manifest.get("supported_concepts", {}).get(plan.concept_id, {}).get("association_layer")
    return {associated} if associated else set()


def _registered_fields(
    query: str, match: re.Match[str], plan: QueryPlan, manifest: dict, policy: _EvidencePolicy
) -> list[str]:
    layers = _relevant_layers(query, match, plan, manifest)
    fields = {
        str(field)
        for layer_id, details in manifest.get("vector_layers", {}).items()
        if layer_id in layers
        for field in details.get("fields", [])
    }
    return sorted(field for field in fields if field.casefold() in policy.fields)


def _enforce_evidence_policy(query: str, plan: QueryPlan, manifest: dict) -> None:
    problems = []
    for policy in _EVIDENCE_POLICIES:
        for match in policy.pattern.finditer(query):
            if _condition_preserves(query, match, policy, plan):
                continue
            fields = _registered_fields(query, match, plan, manifest, policy)
            quoted = match.group(0).strip()
            if fields:
                problems.append(
                    f"'{quoted}' needs {policy.evidence}. Compatible fields are registered "
                    f"({', '.join(fields)}), but this condition was not preserved as a field=value filter"
                )
            else:
                problems.append(
                    f"'{quoted}' needs {policy.evidence}; no compatible field is registered for this dataset"
                )
            break
    if problems:
        raise PlanError(
            "The request contains conditions that RGB imagery or object grounding cannot establish: "
            + "; ".join(problems) + "."
        )


def _reject_unplanned_spatial_language(query: str, manifest: dict) -> None:
    relation = _UNPLANNED_SPATIAL.search(query)
    if relation:
        raise PlanError(
            f"The spatial condition '{relation.group(0)}' is not an executable metric relationship. "
            "Use an explicit 'within N metres of' or 'intersects' clause with a registered layer."
        )
    aliases = {
        str(alias)
        for details in manifest.get("vector_layers", {}).values()
        for alias in details.get("aliases", [])
        if str(alias).strip()
    }
    for alias in sorted(aliases, key=len, reverse=True):
        match = re.search(rf"\b(?:in|on|at|by|inside|outside|along|around)\s+(?:the\s+)?{re.escape(alias)}\b", query, re.I)
        if match:
            raise PlanError(
                f"The spatial phrase '{match.group(0)}' would not be preserved. "
                "Use an explicit 'within N metres of' or 'intersects' clause."
            )


def discovery_plan(query: str, manifest: dict) -> tuple[QueryPlan, str]:
    """Route arbitrary visible descriptions; never drop unexecutable conditions."""
    _reject_unsupported_claims(query)
    if re.search(r"\b(?:constructed|built|construction|between)\b.*\b(?:19|20)\d{2}\b", query, re.I):
        raise PlanError("Construction dates require historical imagery or dated construction records; neither is registered in this sample.")
    _reject_unplanned_spatial_language(query, manifest)
    imagery = _IMAGERY.fullmatch(query)
    if imagery:
        text = imagery.group(1).strip()
        if re.search(r"\b(?:within|intersecting|intersects|near|crossing)\b", text, re.I):
            raise PlanError("Image similarity cannot verify an object-level spatial relationship. Request the objects and relationship directly.")
        plan = validate_plan(QueryPlan(query_type="tiles", concept_id=None, visual_query=text,
            reference_layer=None, predicate=None, distance_m=None), manifest, "explore")
        _enforce_evidence_policy(query, plan, manifest)
        return plan, "explore"
    try:
        return _rule_plan(query, manifest, None, "catalog_scan"), "catalog_scan"
    except PlanError:
        # A new appearance description is grounded by the open-vocabulary model;
        # the same parser still requires every spatial reference to be registered.
        try:
            plan = _rule_plan(query, manifest, None, "ground")
            _enforce_evidence_policy(query, plan, manifest)
            return plan, "ground"
        except PlanError as exc:
            layers = ", ".join(manifest.get("vector_layers", {})) or "none"
            raise PlanError(f"The request cannot be executed against the available evidence: {exc} Available spatial layers: {layers}.") from exc
