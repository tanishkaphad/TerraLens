"""Deterministic dataset/coordinate guards independent of a language model.

These checks preserve explicit constraints. They do not prove that every natural
language clause has been interpreted correctly; semantic review still matters.
History must come from the server's successful conversation records.
"""
from __future__ import annotations

import math
import re

from .workflow import Workflow, WorkflowError, validate_workflow

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_BOX_LABEL = r"(?:bbox|bounding\s*box|wgs\s*84\s*(?:bounds|bbox|coordinates)?|bounds|west\s*[,/]\s*south\s*[,/]\s*east\s*[,/]\s*north)"
_BOX = re.compile(rf"\b{_BOX_LABEL}\s*(?:[:=]|of|is)?\s*[\[(]?\s*({_NUMBER})\s*[,;]\s*({_NUMBER})\s*[,;]\s*({_NUMBER})\s*[,;]\s*({_NUMBER})\s*[\])]?", re.I)
_NAMED = re.compile(rf"\b(west|south|east|north)\s*(?:[:=]|is)\s*({_NUMBER})\b", re.I)


def _valid_box(values) -> tuple[float, float, float, float]:
    box = tuple(float(value) for value in values)
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise WorkflowError("Explicit WGS84 coordinates must contain four finite numeric values.")
    west, south, east, north = box
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise WorkflowError("Explicit WGS84 coordinates must use west,south,east,north with positive extent.")
    return box


def _matches(first, second) -> bool:
    return len(first) == len(second) == 4 and all(math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-9)
        for left, right in zip(first, second))


def explicit_wgs84_boxes(message: str) -> list[tuple[float, float, float, float]]:
    """Read labelled WGS84 boxes; never geocode place names or infer axis order."""
    boxes = [_valid_box(match.groups()) for match in _BOX.finditer(message)]
    named = list(_NAMED.finditer(message))
    if named:
        if len(named) % 4:
            raise WorkflowError("An explicitly named area needs all four west,south,east,north coordinates.")
        for offset in range(0, len(named), 4):
            values = {match.group(1).lower(): match.group(2) for match in named[offset:offset + 4]}
            if set(values) != {"west", "south", "east", "north"}:
                raise WorkflowError("Explicit area coordinates must name west,south,east,north once each.")
            boxes.append(_valid_box([values[key] for key in ("west", "south", "east", "north")]))
    unique = []
    for values in boxes:
        if not any(_matches(values, existing) for existing in unique):
            unique.append(values)
    return unique


def _prior_boxes(history: list, catalog: dict) -> dict[str, list[tuple]]:
    """Carry only coordinate constraints already executed by validated history."""
    authorised = {}
    for turn in history[-8:]:
        if not isinstance(turn, dict) or not isinstance(turn.get("message"), str):
            continue
        result = turn.get("result", {})
        if not isinstance(result, dict) or result.get("status") not in (None, "completed"):
            continue
        try:
            prior = validate_workflow(result.get("workflow"), catalog)
            if prior.status != "ready":
                continue
            explicit = explicit_wgs84_boxes(turn["message"])
        except (ValueError, TypeError):
            continue
        next_boxes = {}
        valid = True
        for step in prior.steps:
            if step.operation != "search":
                continue
            source_boxes = explicit or authorised.get(step.dataset_id, [])
            if step.bbox is not None:
                if not any(_matches(step.bbox, allowed) for allowed in source_boxes):
                    valid = False
                    break
                next_boxes.setdefault(step.dataset_id, []).append(tuple(step.bbox))
            else:
                next_boxes.setdefault(step.dataset_id, [])
        if valid:
            authorised.update(next_boxes)
    return authorised


def guard_intent_constraints(message: str, dataset_id: str | None, history: list,
        workflow: Workflow, catalog: dict) -> dict:
    """Reject dataset switches and fabricated/dropped coordinate constraints.

    Coverage inventory may mention other registered datasets. Every executable
    search stays on the selected dataset. A search bbox must be explicitly
    provided by this user message or authorised successful history for that same
    dataset. Current explicit boxes take precedence over prior coordinate boxes.
    """
    if dataset_id is not None and dataset_id not in catalog:
        raise WorkflowError("The explicitly selected dataset is not registered.")
    workflow = validate_workflow(workflow, catalog)
    if workflow.status != "ready":
        return {"status": "not_executed", "selected_dataset": dataset_id,
            "note": "No geographic steps execute for an unsupported or unresolved request."}
    current_boxes = explicit_wgs84_boxes(message)
    historical = _prior_boxes(history, catalog) if not current_boxes else {}
    used_boxes, step_evidence = [], []
    for step in workflow.steps:
        if step.operation == "coverage":
            continue
        if step.operation != "search":
            continue
        if dataset_id is not None and step.dataset_id != dataset_id:
            raise WorkflowError("A search cannot switch away from the explicitly selected dataset. Select the intended dataset before searching it.")
        if current_boxes and step.bbox is None:
            raise WorkflowError("The explicitly requested WGS84 area was dropped from a geographic search.")
        if step.bbox is not None:
            allowed = current_boxes or historical.get(step.dataset_id, [])
            if not any(_matches(step.bbox, values) for values in allowed):
                raise WorkflowError("A search bbox needs explicit user west,south,east,north coordinates or validated prior coordinates on the same dataset; place names cannot authorise invented coordinates.")
            used_boxes.append(tuple(step.bbox))
            step_evidence.append({"step_id": step.id, "dataset_id": step.dataset_id, "bbox": list(step.bbox),
                "coordinate_source": "current_user_message" if current_boxes else "validated_same_dataset_history"})
        else:
            step_evidence.append({"step_id": step.id, "dataset_id": step.dataset_id, "bbox": None,
                "coordinate_source": "registered_dataset_coverage"})
    if any(not any(_matches(values, used) for used in used_boxes) for values in current_boxes):
        raise WorkflowError("An explicitly requested WGS84 bounding box was omitted from the workflow.")
    return {"status": "passed", "selected_dataset": dataset_id, "search_steps": step_evidence,
        "note": "Deterministic checks preserve explicit dataset/coordinate constraints; they do not prove complete natural-language interpretation."}
