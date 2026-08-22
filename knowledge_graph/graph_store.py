"""
In-memory graph storage abstraction for the knowledge-graph world model.

``GraphStore`` is the single source of truth for persistent world state.  VLM
invocations read context from the store through ``context_retrieval`` and
propose mutations through the ``GraphUpdater``; neither component writes to the
store directly outside of the update pipeline.

Historical facts are retained alongside the current believed state instead of
being destructively overwritten.  A fact that is replaced receives
``valid=False`` but is preserved for provenance and debugging.
"""

from __future__ import annotations

import threading
from typing import Iterator

from .schema import Entity, Fact, utc_now


class GraphStore:
    """Thread-safe in-memory storage for entities and facts.

    This implementation uses plain Python dicts protected by a re-entrant lock.
    Replacing the backing store (e.g., with a graph database) only requires
    implementing the same public interface.

    All mutating methods are atomic with respect to the store lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entities: dict[str, Entity] = {}
        # All facts ever recorded, keyed by fact_id.
        self._facts: dict[str, Fact] = {}

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> Entity:
        """Insert *entity* if its ID is not already present.

        Returns the stored entity (the existing one if the ID was already
        known, so callers can detect no-op upserts by comparing objects).
        """
        with self._lock:
            if entity.entity_id not in self._entities:
                self._entities[entity.entity_id] = entity
            return self._entities[entity.entity_id]

    def get_entity(self, entity_id: str) -> Entity | None:
        """Return the entity with *entity_id*, or ``None``."""
        with self._lock:
            return self._entities.get(entity_id)

    def has_entity(self, entity_id: str) -> bool:
        with self._lock:
            return entity_id in self._entities

    def all_entities(self) -> list[Entity]:
        with self._lock:
            return list(self._entities.values())

    def entity_count(self) -> int:
        with self._lock:
            return len(self._entities)

    # ------------------------------------------------------------------
    # Fact operations
    # ------------------------------------------------------------------

    def add_fact(self, fact: Fact) -> Fact:
        """Insert *fact* into the store unconditionally.

        Callers are responsible for invalidating any superseded facts before
        adding a replacement.
        """
        with self._lock:
            self._facts[fact.fact_id] = fact
            return fact

    def get_fact(self, fact_id: str) -> Fact | None:
        with self._lock:
            return self._facts.get(fact_id)

    def invalidate_fact(self, fact_id: str, *, reason: str = "") -> bool:
        """Mark the fact with *fact_id* as no longer valid.

        Returns ``True`` if the fact was found and updated, ``False`` if it
        does not exist or was already invalid.
        """
        with self._lock:
            fact = self._facts.get(fact_id)
            if fact is None or not fact.valid:
                return False
            fact.valid = False
            if reason:
                fact.provenance = (
                    f"{fact.provenance}; invalidated: {reason}" if fact.provenance else f"invalidated: {reason}"
                )
            return True

    def valid_facts(self) -> list[Fact]:
        """Return all currently valid facts."""
        with self._lock:
            return [f for f in self._facts.values() if f.valid]

    def all_facts(self) -> list[Fact]:
        """Return all facts, including invalidated historical ones."""
        with self._lock:
            return list(self._facts.values())

    def facts_for_entity(self, entity_id: str, *, valid_only: bool = True) -> list[Fact]:
        """Return facts whose subject matches *entity_id*."""
        with self._lock:
            return [
                f
                for f in self._facts.values()
                if f.subject == entity_id and (not valid_only or f.valid)
            ]

    def facts_for_relation(
        self, subject: str, predicate: str, obj: str, *, valid_only: bool = True
    ) -> list[Fact]:
        """Return relation facts matching the triple (subject, predicate, obj)."""
        with self._lock:
            return [
                f
                for f in self._facts.values()
                if (
                    f.fact_type == "relation"
                    and f.subject == subject
                    and f.predicate == predicate
                    and f.obj == obj
                    and (not valid_only or f.valid)
                )
            ]

    def fact_count(self, *, valid_only: bool = False) -> int:
        with self._lock:
            if valid_only:
                return sum(1 for f in self._facts.values() if f.valid)
            return len(self._facts)

    # ------------------------------------------------------------------
    # Snapshots (for context retrieval)
    # ------------------------------------------------------------------

    def snapshot(self) -> tuple[list[Entity], list[Fact]]:
        """Return a consistent point-in-time snapshot of entities and valid facts."""
        with self._lock:
            entities = list(self._entities.values())
            facts = [f for f in self._facts.values() if f.valid]
        return entities, facts
