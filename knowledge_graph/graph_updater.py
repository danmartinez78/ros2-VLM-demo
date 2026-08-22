"""
Graph-update parsing, validation, and reconciliation.

This module implements the contract boundary between the VLM and the
knowledge graph:

1. ``parse_graph_updates``   – extract the structured JSON block from raw VLM
                               output and deserialise it into ``GraphUpdate``
                               objects.
2. ``GraphUpdater.apply``    – validate each proposed mutation, reconcile it
                               with current graph state, and commit accepted
                               updates individually in order (best-effort
                               reconciliation: one rejected update does not
                               prevent subsequent updates from being applied).

Design invariants
-----------------
- Rejected updates never prevent normal VLM task output from being used.
- Each committed fact receives a timestamp, confidence, and provenance.
- Superseded attribute facts are invalidated rather than deleted so that
  the historical record is preserved.
- A new relation that duplicates an existing valid relation is a no-op.
- ``add_entity`` is idempotent: re-adding an existing ID is accepted but
  produces no change to stored data.
- Contradictions (e.g., confidence < min_commit_confidence) are rejected
  with a structured reason without touching the graph.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .graph_store import GraphStore
from .prompt_serializer import GRAPH_UPDATES_END, GRAPH_UPDATES_START
from .schema import (
    Entity,
    Fact,
    GraphUpdate,
    GraphUpdateSet,
    SchemaError,
    clamp_confidence,
    is_valid_entity_id,
    utc_now,
    validate_update,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_graph_updates(vlm_output: str) -> GraphUpdateSet:
    """Extract and deserialise graph-update proposals from raw VLM output.

    The function looks for the first ``<<GRAPH_UPDATES_START>>`` /
    ``<<GRAPH_UPDATES_END>>`` block.  If no such block is found, or the
    content cannot be parsed, it returns an empty ``GraphUpdateSet`` so that
    callers need not handle the absence of updates specially.

    The VLM output outside the markers is not modified or consumed here.
    """
    start_marker = GRAPH_UPDATES_START
    end_marker = GRAPH_UPDATES_END

    start_idx = vlm_output.find(start_marker)
    if start_idx == -1:
        return GraphUpdateSet()

    end_idx = vlm_output.find(end_marker, start_idx)
    if end_idx == -1:
        logger.warning("graph_updater: found %s without matching %s", start_marker, end_marker)
        return GraphUpdateSet()

    json_text = vlm_output[start_idx + len(start_marker) : end_idx].strip()
    if not json_text:
        return GraphUpdateSet()

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("graph_updater: failed to parse graph updates JSON: %s", exc)
        return GraphUpdateSet()

    if not isinstance(payload, dict):
        logger.warning("graph_updater: graph updates payload is not a dict")
        return GraphUpdateSet()

    schema_version = payload.get("schema_version", 1)
    raw_updates = payload.get("graph_updates", [])
    if not isinstance(raw_updates, list):
        logger.warning("graph_updater: 'graph_updates' is not a list")
        return GraphUpdateSet()

    updates: list[GraphUpdate] = []
    for idx, item in enumerate(raw_updates):
        if not isinstance(item, dict):
            logger.warning("graph_updater: update[%d] is not a dict, skipping", idx)
            continue
        try:
            confidence = clamp_confidence(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0

        update = GraphUpdate(
            op=str(item.get("op", "")),
            schema_version=schema_version,
            entity_id=str(item.get("entity_id", "")),
            entity_type=str(item.get("entity_type", "unknown")),
            attribute=str(item.get("attribute", "")),
            value=item.get("value"),
            subject=str(item.get("subject", "")),
            predicate=str(item.get("predicate", "")),
            object=str(item.get("object", "")),
            fact_id=str(item.get("fact_id", "")),
            confidence=confidence,
            timestamp=str(item.get("timestamp", utc_now())),
            source=str(item.get("source", "vlm")),
            provenance=str(item.get("provenance", "")),
        )
        updates.append(update)

    return GraphUpdateSet(schema_version=schema_version, updates=updates)


# ---------------------------------------------------------------------------
# Update result types
# ---------------------------------------------------------------------------


@dataclass
class UpdateResult:
    """Outcome of a single proposed graph update."""

    op: str
    accepted: bool
    reason: str = ""
    # The fact(s) committed as a result of this update.
    committed_facts: list[Fact] = field(default_factory=list)
    # The entity committed (for add_entity).
    committed_entity: Entity | None = None


@dataclass
class ApplyResult:
    """Aggregate result of applying a ``GraphUpdateSet``."""

    total: int = 0
    accepted: int = 0
    rejected: int = 0
    results: list[UpdateResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GraphUpdater
# ---------------------------------------------------------------------------


class GraphUpdater:
    """Validates and commits proposed VLM graph mutations to a ``GraphStore``.

    Parameters
    ----------
    store:
        The graph store to commit accepted mutations to.
    min_commit_confidence:
        Proposed updates with confidence below this threshold are rejected.
        Defaults to 0.5.
    require_known_entities:
        When ``True`` (default ``False``), ``set_attribute``, ``add_relation``,
        and ``remove_relation`` are rejected if referenced entity IDs are not
        already present in the graph.  When ``False``, unknown entity IDs
        trigger an implicit ``add_entity`` call so the graph remains
        self-consistent.
    """

    def __init__(
        self,
        store: GraphStore,
        *,
        min_commit_confidence: float = 0.5,
        require_known_entities: bool = False,
    ) -> None:
        self._store = store
        self._min_commit_confidence = clamp_confidence(min_commit_confidence)
        self._require_known_entities = require_known_entities

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, update_set: GraphUpdateSet) -> ApplyResult:
        """Apply all proposed updates and return an aggregate result.

        Updates are applied individually in order (best-effort reconciliation).
        A failure on one update does not prevent subsequent updates from being
        processed.  This is *not* an all-or-nothing transaction: each accepted
        update is committed immediately and independently.
        """
        result = ApplyResult(total=len(update_set.updates))
        for update in update_set.updates:
            outcome = self._apply_single(update)
            result.results.append(outcome)
            if outcome.accepted:
                result.accepted += 1
            else:
                result.rejected += 1
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_single(self, update: GraphUpdate) -> UpdateResult:
        """Validate and apply a single proposed update."""
        # Schema validation (structure only).
        try:
            validate_update(update)
        except SchemaError as exc:
            return UpdateResult(op=update.op, accepted=False, reason=f"schema error: {exc}")

        # Confidence gate.
        if update.confidence < self._min_commit_confidence:
            return UpdateResult(
                op=update.op,
                accepted=False,
                reason=(
                    f"confidence {update.confidence:.3f} below threshold "
                    f"{self._min_commit_confidence:.3f}"
                ),
            )

        op = update.op

        if op == "add_entity":
            return self._op_add_entity(update)
        elif op == "set_attribute":
            return self._op_set_attribute(update)
        elif op == "add_relation":
            return self._op_add_relation(update)
        elif op == "remove_relation":
            return self._op_remove_relation(update)
        elif op == "invalidate_fact":
            return self._op_invalidate_fact(update)
        else:
            # Should have been caught by validate_update, but guard anyway.
            return UpdateResult(op=op, accepted=False, reason=f"unknown op '{op}'")

    def _ensure_entity(self, entity_id: str, entity_type: str = "unknown") -> bool:
        """Ensure *entity_id* exists, optionally creating it.

        Returns ``False`` (should reject) if ``require_known_entities`` is set
        and the entity is unknown.
        """
        if not self._store.has_entity(entity_id):
            if self._require_known_entities:
                return False
            self._store.add_entity(
                Entity(entity_id=entity_id, entity_type=entity_type)
            )
        return True

    def _op_add_entity(self, update: GraphUpdate) -> UpdateResult:
        entity = Entity(
            entity_id=update.entity_id,
            entity_type=update.entity_type,
            created_at=update.timestamp,
        )
        stored = self._store.add_entity(entity)
        if stored is entity:
            return UpdateResult(op="add_entity", accepted=True, committed_entity=stored)
        # ID already existed; idempotent, still accepted.
        return UpdateResult(op="add_entity", accepted=True, reason="entity already existed", committed_entity=stored)

    def _op_set_attribute(self, update: GraphUpdate) -> UpdateResult:
        if not self._ensure_entity(update.entity_id):
            return UpdateResult(
                op="set_attribute",
                accepted=False,
                reason=f"unknown entity '{update.entity_id}' and require_known_entities=True",
            )

        # Invalidate any prior valid attribute fact for this (entity, attribute) pair.
        prior_facts = [
            f
            for f in self._store.facts_for_entity(update.entity_id)
            if f.fact_type == "attribute" and f.predicate == update.attribute and f.valid
        ]
        for prior in prior_facts:
            self._store.invalidate_fact(prior.fact_id, reason=f"superseded by set_attribute at {update.timestamp}")

        # Commit the new attribute fact.
        new_fact = Fact(
            fact_type="attribute",
            subject=update.entity_id,
            predicate=update.attribute,
            obj=update.value,
            confidence=update.confidence,
            timestamp=update.timestamp,
            source=update.source,
            provenance=update.provenance,
        )
        self._store.add_fact(new_fact)

        # Mirror the attribute onto the entity for quick access.
        entity = self._store.get_entity(update.entity_id)
        if entity is not None:
            entity.attributes[update.attribute] = update.value

        return UpdateResult(op="set_attribute", accepted=True, committed_facts=[new_fact])

    def _op_add_relation(self, update: GraphUpdate) -> UpdateResult:
        if not self._ensure_entity(update.subject):
            return UpdateResult(
                op="add_relation",
                accepted=False,
                reason=f"unknown subject entity '{update.subject}' and require_known_entities=True",
            )
        if not self._ensure_entity(update.object):
            return UpdateResult(
                op="add_relation",
                accepted=False,
                reason=f"unknown object entity '{update.object}' and require_known_entities=True",
            )

        # Idempotency: skip if an identical valid relation already exists.
        existing = self._store.facts_for_relation(update.subject, update.predicate, update.object)
        if existing:
            return UpdateResult(
                op="add_relation",
                accepted=True,
                reason="relation already exists (no-op)",
                committed_facts=existing,
            )

        new_fact = Fact(
            fact_type="relation",
            subject=update.subject,
            predicate=update.predicate,
            obj=update.object,
            confidence=update.confidence,
            timestamp=update.timestamp,
            source=update.source,
            provenance=update.provenance,
        )
        self._store.add_fact(new_fact)
        return UpdateResult(op="add_relation", accepted=True, committed_facts=[new_fact])

    def _op_remove_relation(self, update: GraphUpdate) -> UpdateResult:
        facts = self._store.facts_for_relation(update.subject, update.predicate, update.object)
        if not facts:
            return UpdateResult(op="remove_relation", accepted=True, reason="relation not found (no-op)")
        for fact in facts:
            self._store.invalidate_fact(fact.fact_id, reason=f"removed by VLM at {update.timestamp}")
        return UpdateResult(op="remove_relation", accepted=True, committed_facts=facts)

    def _op_invalidate_fact(self, update: GraphUpdate) -> UpdateResult:
        ok = self._store.invalidate_fact(update.fact_id, reason=f"invalidated by VLM at {update.timestamp}")
        if ok:
            return UpdateResult(op="invalidate_fact", accepted=True)
        return UpdateResult(
            op="invalidate_fact",
            accepted=False,
            reason=f"fact_id '{update.fact_id}' not found or already invalid",
        )
