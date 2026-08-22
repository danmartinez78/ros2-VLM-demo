"""
Schema definitions for the knowledge-graph world model.

Schema version: 1

This module defines the data structures for entities, facts (attributes and
relations), and the VLM graph-update contract.  All timestamps are UTC ISO-8601
strings.  Confidence values are floats in [0.0, 1.0].

Allowed graph-update operations
--------------------------------
add_entity       – create a new entity with optional initial attributes
set_attribute    – set or overwrite a named attribute on an entity
add_relation     – add a directed relation (subject --predicate--> object)
remove_relation  – remove a specific relation
invalidate_fact  – mark an attribute or relation as no longer believed true
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1

# Valid operation names in graph_updates payloads.
VALID_OPS: frozenset[str] = frozenset(
    {
        "add_entity",
        "set_attribute",
        "add_relation",
        "remove_relation",
        "invalidate_fact",
    }
)

_ENTITY_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

# Maximum length for entity IDs and string attribute values.
MAX_ID_LENGTH: int = 128
MAX_VALUE_LENGTH: int = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_entity_id(prefix: str = "entity") -> str:
    """Generate a unique entity ID with an optional prefix."""
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{suffix}"


def is_valid_entity_id(entity_id: str) -> bool:
    """Return True if *entity_id* is a well-formed entity identifier."""
    return (
        isinstance(entity_id, str)
        and 1 <= len(entity_id) <= MAX_ID_LENGTH
        and _ENTITY_ID_RE.match(entity_id) is not None
    )


def clamp_confidence(value: Any) -> float:
    """Coerce *value* to a float confidence in [0.0, 1.0].

    Raises ``TypeError`` if the value is not numeric.
    """
    v = float(value)
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Core graph data structures
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A persistent entity in the knowledge graph."""

    entity_id: str
    entity_type: str = "unknown"
    created_at: str = field(default_factory=utc_now)
    # Free-form attributes stored directly on the entity for quick access.
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    """A single ground fact: either an attribute or a relation.

    ``fact_type`` is either ``"attribute"`` or ``"relation"``.

    For attributes:
      - ``subject``   – entity_id that owns the attribute
      - ``predicate`` – attribute name
      - ``obj``       – attribute value (arbitrary JSON-serialisable type)

    For relations:
      - ``subject``   – source entity_id
      - ``predicate`` – relation label
      - ``obj``       – target entity_id
    """

    fact_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    fact_type: Literal["attribute", "relation"] = "attribute"
    subject: str = ""
    predicate: str = ""
    obj: Any = None
    confidence: float = 1.0
    timestamp: str = field(default_factory=utc_now)
    source: str = "vlm"
    valid: bool = True
    # Original observation text or context that produced this fact (optional).
    provenance: str = ""


# ---------------------------------------------------------------------------
# Graph-update contract (VLM output)
# ---------------------------------------------------------------------------


@dataclass
class GraphUpdate:
    """A single proposed graph mutation returned by the VLM."""

    op: str
    schema_version: int = SCHEMA_VERSION
    # Fields used by different operations (not all are required for every op):
    entity_id: str = ""
    entity_type: str = "unknown"
    attribute: str = ""
    value: Any = None
    subject: str = ""
    predicate: str = ""
    object: str = ""
    fact_id: str = ""
    confidence: float = 1.0
    timestamp: str = field(default_factory=utc_now)
    source: str = "vlm"
    provenance: str = ""


@dataclass
class GraphUpdateSet:
    """The structured portion of a VLM response containing proposed mutations."""

    schema_version: int = SCHEMA_VERSION
    updates: list[GraphUpdate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when a graph-update fails schema validation."""


def validate_update(update: GraphUpdate) -> None:  # noqa: C901  (acceptable complexity)
    """Raise ``SchemaError`` if *update* does not satisfy the schema.

    This validates structure only; it does not check entity existence.
    """
    if update.op not in VALID_OPS:
        raise SchemaError(f"Unknown operation '{update.op}'; allowed: {sorted(VALID_OPS)}")

    op = update.op

    if op == "add_entity":
        if not is_valid_entity_id(update.entity_id):
            raise SchemaError(
                f"add_entity requires a valid entity_id; got {update.entity_id!r}"
            )

    elif op == "set_attribute":
        if not is_valid_entity_id(update.entity_id):
            raise SchemaError(
                f"set_attribute requires a valid entity_id; got {update.entity_id!r}"
            )
        if not update.attribute:
            raise SchemaError("set_attribute requires a non-empty 'attribute'")
        if update.value is None:
            raise SchemaError("set_attribute requires a non-None 'value'")
        if isinstance(update.value, str) and len(update.value) > MAX_VALUE_LENGTH:
            raise SchemaError(
                f"set_attribute value exceeds max length {MAX_VALUE_LENGTH}"
            )

    elif op == "add_relation":
        if not is_valid_entity_id(update.subject):
            raise SchemaError(
                f"add_relation requires a valid 'subject' entity_id; got {update.subject!r}"
            )
        if not is_valid_entity_id(update.object):
            raise SchemaError(
                f"add_relation requires a valid 'object' entity_id; got {update.object!r}"
            )
        if not update.predicate:
            raise SchemaError("add_relation requires a non-empty 'predicate'")

    elif op == "remove_relation":
        if not is_valid_entity_id(update.subject):
            raise SchemaError(
                f"remove_relation requires a valid 'subject' entity_id; got {update.subject!r}"
            )
        if not is_valid_entity_id(update.object):
            raise SchemaError(
                f"remove_relation requires a valid 'object' entity_id; got {update.object!r}"
            )
        if not update.predicate:
            raise SchemaError("remove_relation requires a non-empty 'predicate'")

    elif op == "invalidate_fact":
        if not update.fact_id:
            raise SchemaError("invalidate_fact requires a non-empty 'fact_id'")

    try:
        update.confidence = clamp_confidence(update.confidence)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Invalid confidence value: {exc}") from exc
