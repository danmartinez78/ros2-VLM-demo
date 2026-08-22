"""
Compact, deterministic VLM prompt serialisation for a retrieved subgraph.

The serialiser converts a ``SubGraph`` into a structured JSON-like text block
suitable for inclusion in a VLM system or user prompt.  It preserves:

- stable entity IDs
- entity types and current attributes
- relation facts with predicate, object, confidence, and timestamp
- attribute facts with value, confidence, and timestamp

The serialised block also includes instructions that tell the VLM how to
return structured graph-update proposals alongside its normal output, so that
the update parser can extract them.

The output schema version is embedded in the block for forward compatibility.
"""

from __future__ import annotations

import json
from typing import Any

from .context_retrieval import SubGraph
from .schema import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Prompt block markers (stable across versions)
# ---------------------------------------------------------------------------

GRAPH_CONTEXT_START = "<<GRAPH_CONTEXT_START>>"
GRAPH_CONTEXT_END = "<<GRAPH_CONTEXT_END>>"
GRAPH_UPDATES_START = "<<GRAPH_UPDATES_START>>"
GRAPH_UPDATES_END = "<<GRAPH_UPDATES_END>>"


# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------


def serialize_subgraph(subgraph: SubGraph, *, include_scores: bool = False) -> str:
    """Serialise *subgraph* to a compact JSON block for the VLM prompt.

    The returned string is a self-contained block bounded by
    ``GRAPH_CONTEXT_START`` / ``GRAPH_CONTEXT_END`` markers.

    Parameters
    ----------
    subgraph:
        The subgraph produced by ``retrieve_salient_subgraph``.
    include_scores:
        When ``True``, include salience scores in the output (useful for
        debugging).
    """
    entities_out: list[dict[str, Any]] = []
    for entity in subgraph.entities:
        entry: dict[str, Any] = {
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
        }
        if entity.attributes:
            entry["attributes"] = entity.attributes
        if include_scores and entity.entity_id in subgraph.scores:
            entry["salience_score"] = subgraph.scores[entity.entity_id]
        entities_out.append(entry)

    relations_out: list[dict[str, Any]] = []
    attributes_out: list[dict[str, Any]] = []

    for fact in subgraph.facts:
        if fact.fact_type == "relation":
            relations_out.append(
                {
                    "fact_id": fact.fact_id,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.obj,
                    "confidence": round(fact.confidence, 4),
                    "timestamp": fact.timestamp,
                    "source": fact.source,
                }
            )
        else:
            attributes_out.append(
                {
                    "fact_id": fact.fact_id,
                    "entity_id": fact.subject,
                    "attribute": fact.predicate,
                    "value": fact.obj,
                    "confidence": round(fact.confidence, 4),
                    "timestamp": fact.timestamp,
                    "source": fact.source,
                }
            )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "entities": entities_out,
        "attribute_facts": attributes_out,
        "relation_facts": relations_out,
    }

    lines = [
        GRAPH_CONTEXT_START,
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        GRAPH_CONTEXT_END,
    ]
    return "\n".join(lines)


def build_update_instructions() -> str:
    """Return the standing instructions telling the VLM how to propose graph updates.

    This text is intended to be included in the system prompt once so that all
    invocations share the same contract without repeating it in every user turn.
    """
    example = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "graph_updates": [
                {
                    "op": "set_attribute",
                    "entity_id": "dog_17",
                    "attribute": "last_seen_location",
                    "value": "parking_lot_1",
                    "confidence": 0.91,
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "source": "vlm",
                    "provenance": "visible in current frame",
                },
                {
                    "op": "add_relation",
                    "subject": "dog_17",
                    "predicate": "near",
                    "object": "vehicle_12",
                    "confidence": 0.84,
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "source": "vlm",
                    "provenance": "spatial proximity in frame",
                },
            ],
        },
        indent=2,
    )
    return (
        "After your response, if you observed any facts that should update the world model, "
        f"output them between {GRAPH_UPDATES_START} and {GRAPH_UPDATES_END} markers as JSON "
        f"(schema_version={SCHEMA_VERSION}).\n"
        "Allowed ops: add_entity, set_attribute, add_relation, remove_relation, invalidate_fact.\n"
        "Only propose updates for observations you are confident about.\n"
        "If there are no updates, omit the block entirely.\n\n"
        f"Example:\n{GRAPH_UPDATES_START}\n{example}\n{GRAPH_UPDATES_END}"
    )
