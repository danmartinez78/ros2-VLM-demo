"""
Salient subgraph retrieval for VLM context.

Before each VLM invocation, this module selects the most relevant portion of
the knowledge graph (bounded by a configurable token/character budget) and
returns it for serialisation into the prompt.

Scoring model
-------------
Each entity is scored by a weighted sum of signals:

    score(entity) =
        task_relevance_weight   * task_relevance(entity)
      + entity_relevance_weight * entity_relevance(entity, mention_ids)
      + recency_weight          * recency(entity, current_time)
      + confidence_weight       * mean_confidence(entity)

Relation facts are included if at least one of their endpoint entities is in
the selected entity set.

The scoring weights are configurable via ``RetrievalConfig``.  The ranking
strategy can evolve without changing the downstream serialisation or VLM
integration contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .graph_store import GraphStore
from .schema import Entity, Fact


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    """Weights and limits for salient subgraph retrieval.

    ``context_budget`` is measured in approximate characters.  The serialiser
    uses this as a soft upper bound and stops adding entities once the budget
    is exhausted.
    """

    context_budget: int = 4000  # approx characters

    # Scoring weights (all non-negative floats; need not sum to 1).
    task_relevance_weight: float = 3.0
    entity_relevance_weight: float = 2.0
    recency_weight: float = 1.0
    confidence_weight: float = 1.0

    # Minimum confidence threshold; facts below this are excluded.
    min_confidence: float = 0.0

    # Maximum number of entities to include regardless of budget.
    max_entities: int = 50


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(ts: str) -> float:
    """Parse an ISO-8601 timestamp string to a POSIX timestamp (float seconds)."""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _recency_score(ts: float | str, now_ts: float, half_life_seconds: float = 300.0) -> float:
    """Exponential decay score: 1.0 when fresh, decaying toward 0 over time.

    *ts* may be either a POSIX timestamp (float) or an ISO-8601 string.
    """
    if isinstance(ts, str):
        ts = _parse_timestamp(ts)
    age = max(0.0, now_ts - ts)
    return math.exp(-age / half_life_seconds)


def _entity_label(entity: Entity) -> str:
    """Return a display label for an entity (type + ID)."""
    return f"{entity.entity_type}:{entity.entity_id}"


# ---------------------------------------------------------------------------
# Subgraph result
# ---------------------------------------------------------------------------


@dataclass
class SubGraph:
    """A bounded, scored subgraph ready for serialisation into a VLM prompt."""

    entities: list[Entity] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    # Scores keyed by entity_id for debugging / observability.
    scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval function
# ---------------------------------------------------------------------------


def retrieve_salient_subgraph(
    store: GraphStore,
    *,
    task_keywords: list[str] | None = None,
    mentioned_entity_ids: list[str] | None = None,
    config: RetrievalConfig | None = None,
    now_ts: float | None = None,
) -> SubGraph:
    """Select the most salient subgraph for the current observation/task.

    Parameters
    ----------
    store:
        The graph store to read from.
    task_keywords:
        Keywords extracted from the current task or query.  Entities whose
        type, ID, or attribute values contain any keyword receive a task-
        relevance boost.
    mentioned_entity_ids:
        Entity IDs explicitly detected or referenced in the current observation
        (e.g., from an upstream tracker).
    config:
        Retrieval configuration.  Defaults to ``RetrievalConfig()``.
    now_ts:
        Current POSIX timestamp (float seconds).  Defaults to ``time.time()``.
    """
    if config is None:
        config = RetrievalConfig()
    if now_ts is None:
        import time

        now_ts = time.time()

    mentioned_ids: frozenset[str] = frozenset(mentioned_entity_ids or [])
    kw_lower: list[str] = [k.lower() for k in (task_keywords or [])]

    entities, facts = store.snapshot()

    # Index valid facts by subject entity for quick lookup.
    facts_by_subject: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.confidence >= config.min_confidence:
            facts_by_subject.setdefault(fact.subject, []).append(fact)

    # ------------------------------------------------------------------
    # Score each entity
    # ------------------------------------------------------------------
    scored: list[tuple[float, Entity]] = []

    for entity in entities:
        entity_facts = facts_by_subject.get(entity.entity_id, [])

        # Task relevance: does any task keyword appear in ID / type / attributes?
        task_rel = 0.0
        if kw_lower:
            searchable = (
                entity.entity_id.lower()
                + " "
                + entity.entity_type.lower()
                + " "
                + " ".join(str(v).lower() for v in entity.attributes.values())
            )
            matches = sum(1 for kw in kw_lower if kw in searchable)
            task_rel = matches / len(kw_lower)

        # Entity relevance: was this entity explicitly mentioned?
        entity_rel = 1.0 if entity.entity_id in mentioned_ids else 0.0

        # Recency: use the most recent valid fact timestamp (or entity creation).
        if entity_facts:
            latest_ts = max(_parse_timestamp(f.timestamp) for f in entity_facts)
        else:
            latest_ts = _parse_timestamp(entity.created_at)
        recency = _recency_score(latest_ts, now_ts)

        # Confidence: mean confidence of valid facts for this entity.
        mean_conf = (
            sum(f.confidence for f in entity_facts) / len(entity_facts) if entity_facts else 0.5
        )

        score = (
            config.task_relevance_weight * task_rel
            + config.entity_relevance_weight * entity_rel
            + config.recency_weight * recency
            + config.confidence_weight * mean_conf
        )
        scored.append((score, entity))

    # Sort descending by score, then by entity_id for determinism.
    scored.sort(key=lambda t: (-t[0], t[1].entity_id))

    # ------------------------------------------------------------------
    # Select entities within budget
    # ------------------------------------------------------------------
    selected_entities: list[Entity] = []
    selected_ids: set[str] = set()
    scores_out: dict[str, float] = {}
    budget_remaining = config.context_budget

    for score, entity in scored:
        if len(selected_entities) >= config.max_entities:
            break
        # Rough cost estimate: ID + type + attributes as JSON.
        cost = len(entity.entity_id) + len(entity.entity_type) + sum(
            len(str(k)) + len(str(v)) for k, v in entity.attributes.items()
        ) + 20  # overhead
        if budget_remaining <= 0 and selected_entities:
            # Always include at least one entity even if over budget.
            break
        selected_entities.append(entity)
        selected_ids.add(entity.entity_id)
        scores_out[entity.entity_id] = round(score, 4)
        budget_remaining -= cost

    # ------------------------------------------------------------------
    # Collect facts whose subject is in the selected entity set.
    # Also include relation facts where the object is selected.
    # ------------------------------------------------------------------
    selected_facts: list[Fact] = []
    for fact in facts:
        if fact.confidence < config.min_confidence:
            continue
        if fact.subject in selected_ids:
            selected_facts.append(fact)
        elif fact.fact_type == "relation" and isinstance(fact.obj, str) and fact.obj in selected_ids:
            selected_facts.append(fact)

    # Sort facts for deterministic output.
    selected_facts.sort(key=lambda f: (f.subject, f.predicate, f.fact_id))

    return SubGraph(entities=selected_entities, facts=selected_facts, scores=scores_out)
