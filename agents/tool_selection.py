"""Model-directed tool selection.

Why this module exists
-----------------------
Agents used to pick tools via hardcoded Python name lists (``_pick_tool`` with
``SEARCH_TOOL_NAMES`` etc.) and ``inspect.signature`` arg binding. No
function-calling schema was ever sent to a model: the "choice" of tool was a
``dict.get`` in preference order, invisible in the trace.

Here every tool use is preceded by a model decision:

* :func:`function_schema_for_tool` renders a tool's ``run`` signature as an
  OpenAI-style function definition (``{name, description, parameters}``), so
  the model sees what each tool actually takes;
* :func:`select_tool` filters ``ctx.tools`` to the candidates with a callable
  ``run``, builds those schemas, and issues a ``CHOICE`` ``DecisionRequest``
  whose ``state`` carries ``purpose``, ``available_tools``, ``tool_schemas``
  and ``candidate_count``. The decision layer forwards ``tool_schemas``
  verbatim (see ``decision/clef.py:build_body`` and
  ``decision/gemini.py:build_prompt``).

Fallbacks keep old behaviour identical: no candidates means ``(None, None,
None)`` without calling ``decide``; an unreachable backend
(``DecisionUnavailable``/``DecisionFailed``) or an off-menu answer falls back
to the first candidate. A single candidate is offered as ``[name,
"skip_tool"]`` because a ``CHOICE`` needs at least two options.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any

from core.errors import DecisionFailed, DecisionUnavailable
from core.ids import new_id
from core.schemas import (
    Decision,
    DecisionRequest,
    DecisionSource,
    QuestionType,
)

__all__ = [
    "function_schema_for_tool",
    "describe_tools_for_model",
    "select_tool",
]

#: Offered alongside a lone candidate so a ``CHOICE`` still has two options.
SKIP_OPTION = "skip_tool"


def _json_type(annotation: Any) -> str:
    """Map a Python annotation to a JSON-schema type name."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "string"
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is type(None):
        return "null"
    origin = typing.get_origin(annotation)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
        return "string"
    if origin in (list, tuple, set, frozenset):
        return "array"
    if origin in (dict,):
        return "object"
    try:
        if isinstance(annotation, type):
            if issubclass(annotation, str):
                return "string"
            if issubclass(annotation, bool):
                return "boolean"
            if issubclass(annotation, int):
                return "integer"
            if issubclass(annotation, float):
                return "number"
            if issubclass(annotation, (list, tuple, set, frozenset)):
                return "array"
            if issubclass(annotation, dict):
                return "object"
    except TypeError:
        pass
    annotation_str = str(annotation)
    if "Sequence" in annotation_str or "List" in annotation_str:
        return "array"
    if "Mapping" in annotation_str or "Dict" in annotation_str:
        return "object"
    if "int" in annotation_str.lower():
        return "integer"
    if "float" in annotation_str.lower() or "number" in annotation_str.lower():
        return "number"
    if "bool" in annotation_str.lower():
        return "boolean"
    return "string"


def function_schema_for_tool(tool: Any) -> dict[str, Any]:
    """Render ``tool.run`` as an OpenAI-style function definition.

    Returns ``{name, description, parameters: {type, properties, required,
    additionalProperties}}`` where ``properties`` comes from
    ``inspect.signature(tool.run)``. A ``**kwargs`` parameter maps to
    ``additionalProperties: True``; otherwise ``False``.
    """
    name = str(getattr(tool, "name", None) or type(tool).__name__)
    description = str(getattr(tool, "description", "") or "")
    if not description:
        doc = inspect.getdoc(getattr(tool, "run", tool)) or ""
        description = doc.strip().splitlines()[0] if doc.strip() else name
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    additional_properties = False
    runner = getattr(tool, "run", None)
    try:
        signature = inspect.signature(runner) if callable(runner) else None
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for param in signature.parameters.values():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                additional_properties = True
                continue
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            properties[param.name] = {"type": _json_type(param.annotation)}
            if param.default is inspect.Parameter.empty:
                required.append(param.name)
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": additional_properties,
        },
    }


def describe_tools_for_model(tools: Any) -> list[dict[str, Any]]:
    """One function schema per tool, for inclusion in a decision ``state``."""
    if isinstance(tools, Mapping):
        items = list(tools.values())
    elif isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
        items = list(tools)
    else:
        try:
            items = list(tools)
        except TypeError:
            items = [tools]
    return [function_schema_for_tool(tool) for tool in items]


def select_tool(
    ctx: Any,
    purpose: str,
    candidate_names: Sequence[str],
    decision_point: str,
    instructions: str = "",
) -> tuple[Any | None, str | None, Decision | None]:
    """Ask the model which registered tool to use for ``purpose``.

    Filters ``ctx.tools`` to the ``candidate_names`` with a callable ``run``,
    builds their function schemas, and issues a ``CHOICE`` ``DecisionRequest``
    with ``state`` containing ``purpose``, ``available_tools``,
    ``tool_schemas`` and ``candidate_count``.

    Returns ``(tool, name, decision)``. No candidates means ``(None, None,
    None)`` without calling ``decide``. An unreachable backend or an off-menu
    answer falls back to the first candidate. A lone candidate is offered as
    ``[name, "skip_tool"]``; an explicit ``skip_tool`` choice returns
    ``(None, None, decision)``.
    """
    _ = DecisionSource  # re-exported for callers grepping the decision path
    tools_map: Mapping[str, Any] = getattr(ctx, "tools", None) or {}
    candidates: list[tuple[str, Any]] = []
    for candidate_name in list(candidate_names or []):
        try:
            tool = tools_map.get(candidate_name)
        except AttributeError:
            tool = None
        if tool is not None and callable(getattr(tool, "run", None)):
            candidates.append((candidate_name, tool))
    if not candidates:
        return None, None, None

    available = [name for name, _ in candidates]
    schemas = describe_tools_for_model([tool for _, tool in candidates])
    options = ([available[0], SKIP_OPTION] if len(candidates) == 1
               else list(available))

    question = f"Which tool should be used for {purpose}?" if purpose else (
        "Which of the available tools should be used for this step?"
    )
    effective_instructions = instructions.strip() or (
        f"Choose exactly one tool to achieve: {purpose or 'the current step'}. "
        "Answer with exactly one of the offered options."
    )
    asked_by = getattr(ctx, "agent", None)
    try:
        from core.schemas import AgentId as _AgentId

        if not isinstance(asked_by, _AgentId):
            asked_by = None
    except ImportError:  # pragma: no cover - core.schemas is always present
        asked_by = None

    request = DecisionRequest(
        request_id=new_id("dec"),
        question=question,
        question_type=QuestionType.CHOICE,
        state={
            "purpose": purpose,
            "available_tools": available,
            "tool_schemas": schemas,
            "candidate_count": len(candidates),
        },
        options=options,
        instructions=effective_instructions,
        asked_by=asked_by,
        decision_point=decision_point,
    )
    try:
        decision = ctx.decide(request)
    except (DecisionUnavailable, DecisionFailed):
        first_name, first_tool = candidates[0]
        return first_tool, first_name, None
    if not isinstance(decision, Decision):
        first_name, first_tool = candidates[0]
        return first_tool, first_name, None
    choice = (decision.choice or "").strip()
    if choice == SKIP_OPTION:
        return None, None, decision
    for candidate_name, tool in candidates:
        if candidate_name == choice:
            return tool, candidate_name, decision
    first_name, first_tool = candidates[0]
    return first_tool, first_name, decision
