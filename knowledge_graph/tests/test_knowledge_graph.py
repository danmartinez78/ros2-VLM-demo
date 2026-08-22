"""
CI-safe tests for the knowledge-graph world model.

These tests do NOT require TensorRT, CUDA, ROS, or any hardware.
They validate:
  - schema validation for all graph-update operations
  - GraphStore entity and fact operations (add, invalidate, snapshot)
  - salient subgraph retrieval (scoring, budget, determinism)
  - VLM prompt serialisation (markers, JSON structure)
  - graph-update parsing from raw VLM output (happy path and malformed inputs)
  - GraphUpdater reconciliation (accepted, rejected, idempotent cases)
  - full round-trip: graph -> context -> VLM output -> validated graph updates -> graph
  - edge cases: malformed updates, contradictory updates, low-confidence updates,
    empty/no-op updates
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

# Ensure repo root is importable without installation.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from knowledge_graph.schema import (
    SCHEMA_VERSION,
    Entity,
    Fact,
    GraphUpdate,
    GraphUpdateSet,
    SchemaError,
    clamp_confidence,
    is_valid_entity_id,
    new_entity_id,
    utc_now,
    validate_update,
)
from knowledge_graph.graph_store import GraphStore
from knowledge_graph.context_retrieval import RetrievalConfig, SubGraph, retrieve_salient_subgraph
from knowledge_graph.prompt_serializer import (
    GRAPH_CONTEXT_END,
    GRAPH_CONTEXT_START,
    GRAPH_UPDATES_END,
    GRAPH_UPDATES_START,
    build_update_instructions,
    serialize_subgraph,
)
from knowledge_graph.graph_updater import (
    ApplyResult,
    GraphUpdater,
    UpdateResult,
    parse_graph_updates,
)


# ===========================================================================
# Schema tests
# ===========================================================================


class TestIsValidEntityId(unittest.TestCase):
    def test_valid_ids(self):
        for eid in ["dog_17", "vehicle-12", "EntityABC", "e1", "a" * 128]:
            self.assertTrue(is_valid_entity_id(eid), eid)

    def test_invalid_ids(self):
        for eid in ["", "a" * 129, "dog 17", "dog/17", "dog@17", None, 123]:
            self.assertFalse(is_valid_entity_id(eid), eid)


class TestClampConfidence(unittest.TestCase):
    def test_clamping(self):
        self.assertEqual(clamp_confidence(0.5), 0.5)
        self.assertEqual(clamp_confidence(-1.0), 0.0)
        self.assertEqual(clamp_confidence(2.0), 1.0)
        self.assertEqual(clamp_confidence("0.8"), 0.8)

    def test_non_numeric_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            clamp_confidence("abc")


class TestValidateUpdate(unittest.TestCase):
    def _make(self, **kwargs) -> GraphUpdate:
        defaults = {
            "op": "add_entity",
            "entity_id": "dog_17",
            "confidence": 0.9,
        }
        defaults.update(kwargs)
        return GraphUpdate(**defaults)

    # add_entity
    def test_add_entity_valid(self):
        validate_update(self._make(op="add_entity", entity_id="dog_17"))

    def test_add_entity_bad_id(self):
        with self.assertRaises(SchemaError):
            validate_update(self._make(op="add_entity", entity_id=""))

    def test_add_entity_invalid_id_chars(self):
        with self.assertRaises(SchemaError):
            validate_update(self._make(op="add_entity", entity_id="dog 17"))

    # set_attribute
    def test_set_attribute_valid(self):
        validate_update(self._make(op="set_attribute", entity_id="dog_17", attribute="color", value="brown"))

    def test_set_attribute_missing_attribute(self):
        with self.assertRaises(SchemaError):
            validate_update(self._make(op="set_attribute", entity_id="dog_17", attribute="", value="x"))

    def test_set_attribute_missing_value(self):
        with self.assertRaises(SchemaError):
            validate_update(self._make(op="set_attribute", entity_id="dog_17", attribute="color", value=None))

    def test_set_attribute_value_too_long(self):
        with self.assertRaises(SchemaError):
            validate_update(self._make(op="set_attribute", entity_id="dog_17", attribute="desc", value="x" * 1025))

    # add_relation
    def test_add_relation_valid(self):
        validate_update(GraphUpdate(op="add_relation", subject="dog_17", predicate="near", object="vehicle_12", confidence=0.8))

    def test_add_relation_bad_subject(self):
        with self.assertRaises(SchemaError):
            validate_update(GraphUpdate(op="add_relation", subject="", predicate="near", object="vehicle_12"))

    def test_add_relation_bad_object(self):
        with self.assertRaises(SchemaError):
            validate_update(GraphUpdate(op="add_relation", subject="dog_17", predicate="near", object=""))

    def test_add_relation_empty_predicate(self):
        with self.assertRaises(SchemaError):
            validate_update(GraphUpdate(op="add_relation", subject="dog_17", predicate="", object="vehicle_12"))

    # remove_relation
    def test_remove_relation_valid(self):
        validate_update(GraphUpdate(op="remove_relation", subject="dog_17", predicate="near", object="vehicle_12", confidence=0.9))

    # invalidate_fact
    def test_invalidate_fact_valid(self):
        validate_update(GraphUpdate(op="invalidate_fact", fact_id="abc123", confidence=1.0))

    def test_invalidate_fact_empty_id(self):
        with self.assertRaises(SchemaError):
            validate_update(GraphUpdate(op="invalidate_fact", fact_id="", confidence=1.0))

    # unknown op
    def test_unknown_op(self):
        with self.assertRaises(SchemaError):
            validate_update(GraphUpdate(op="delete_everything"))

    # bad confidence
    def test_bad_confidence_non_numeric(self):
        with self.assertRaises(SchemaError):
            validate_update(GraphUpdate(op="add_entity", entity_id="e1", confidence="bad"))


# ===========================================================================
# GraphStore tests
# ===========================================================================


class TestGraphStore(unittest.TestCase):
    def setUp(self):
        self.store = GraphStore()

    def test_add_and_get_entity(self):
        e = Entity(entity_id="dog_17", entity_type="dog")
        self.store.add_entity(e)
        self.assertIs(self.store.get_entity("dog_17"), e)

    def test_add_entity_idempotent(self):
        e1 = Entity(entity_id="dog_17", entity_type="dog")
        e2 = Entity(entity_id="dog_17", entity_type="canine")
        stored1 = self.store.add_entity(e1)
        stored2 = self.store.add_entity(e2)
        self.assertIs(stored1, stored2)  # second add returns existing
        self.assertEqual(stored1.entity_type, "dog")

    def test_get_unknown_entity_returns_none(self):
        self.assertIsNone(self.store.get_entity("unknown"))

    def test_has_entity(self):
        self.store.add_entity(Entity(entity_id="e1"))
        self.assertTrue(self.store.has_entity("e1"))
        self.assertFalse(self.store.has_entity("e99"))

    def test_add_and_get_fact(self):
        f = Fact(fact_type="attribute", subject="dog_17", predicate="color", obj="brown", confidence=0.9)
        self.store.add_entity(Entity(entity_id="dog_17"))
        self.store.add_fact(f)
        self.assertIs(self.store.get_fact(f.fact_id), f)

    def test_invalidate_fact(self):
        f = Fact(fact_type="attribute", subject="e1", predicate="x", obj=1)
        self.store.add_fact(f)
        self.assertTrue(f.valid)
        result = self.store.invalidate_fact(f.fact_id)
        self.assertTrue(result)
        self.assertFalse(f.valid)

    def test_invalidate_unknown_fact_returns_false(self):
        self.assertFalse(self.store.invalidate_fact("nonexistent_id"))

    def test_invalidate_already_invalid_returns_false(self):
        f = Fact(fact_type="attribute", subject="e1", predicate="x", obj=1)
        self.store.add_fact(f)
        self.store.invalidate_fact(f.fact_id)
        self.assertFalse(self.store.invalidate_fact(f.fact_id))

    def test_valid_facts_excludes_invalidated(self):
        f1 = Fact(fact_type="attribute", subject="e1", predicate="a", obj=1)
        f2 = Fact(fact_type="attribute", subject="e1", predicate="b", obj=2)
        self.store.add_fact(f1)
        self.store.add_fact(f2)
        self.store.invalidate_fact(f1.fact_id)
        valid = self.store.valid_facts()
        self.assertNotIn(f1, valid)
        self.assertIn(f2, valid)

    def test_facts_for_entity(self):
        f1 = Fact(fact_type="attribute", subject="e1", predicate="x", obj=1)
        f2 = Fact(fact_type="attribute", subject="e2", predicate="y", obj=2)
        self.store.add_fact(f1)
        self.store.add_fact(f2)
        facts = self.store.facts_for_entity("e1")
        self.assertEqual(len(facts), 1)
        self.assertIs(facts[0], f1)

    def test_snapshot_consistency(self):
        e = Entity(entity_id="e1")
        f = Fact(fact_type="attribute", subject="e1", predicate="x", obj=42)
        self.store.add_entity(e)
        self.store.add_fact(f)
        entities, facts = self.store.snapshot()
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(facts), 1)

    def test_fact_count(self):
        f1 = Fact(fact_type="attribute", subject="e1", predicate="a", obj=1)
        f2 = Fact(fact_type="attribute", subject="e1", predicate="b", obj=2)
        self.store.add_fact(f1)
        self.store.add_fact(f2)
        self.store.invalidate_fact(f1.fact_id)
        self.assertEqual(self.store.fact_count(), 2)
        self.assertEqual(self.store.fact_count(valid_only=True), 1)


# ===========================================================================
# Context retrieval tests
# ===========================================================================


class TestContextRetrieval(unittest.TestCase):
    def _build_store(self):
        store = GraphStore()
        for i in range(5):
            store.add_entity(Entity(entity_id=f"dog_{i}", entity_type="dog"))
            f = Fact(fact_type="attribute", subject=f"dog_{i}", predicate="location", obj=f"zone_{i}", confidence=0.9)
            store.add_fact(f)
        store.add_entity(Entity(entity_id="vehicle_1", entity_type="vehicle"))
        rel = Fact(fact_type="relation", subject="dog_0", predicate="near", obj="vehicle_1", confidence=0.8)
        store.add_fact(rel)
        return store

    def test_basic_retrieval_returns_subgraph(self):
        store = self._build_store()
        sg = retrieve_salient_subgraph(store)
        self.assertIsInstance(sg, SubGraph)
        self.assertGreater(len(sg.entities), 0)

    def test_mentioned_entity_boosted(self):
        store = self._build_store()
        sg = retrieve_salient_subgraph(store, mentioned_entity_ids=["dog_3"])
        entity_ids = [e.entity_id for e in sg.entities]
        self.assertIn("dog_3", entity_ids)

    def test_task_keywords_boost_relevant_entities(self):
        store = self._build_store()
        sg = retrieve_salient_subgraph(store, task_keywords=["vehicle"])
        entity_ids = [e.entity_id for e in sg.entities]
        self.assertIn("vehicle_1", entity_ids)

    def test_context_budget_limits_entities(self):
        store = self._build_store()
        config = RetrievalConfig(context_budget=5)  # very small
        sg = retrieve_salient_subgraph(store, config=config)
        self.assertGreaterEqual(len(sg.entities), 1)

    def test_max_entities_limit(self):
        store = GraphStore()
        for i in range(20):
            store.add_entity(Entity(entity_id=f"e_{i}"))
        config = RetrievalConfig(max_entities=5)
        sg = retrieve_salient_subgraph(store, config=config)
        self.assertLessEqual(len(sg.entities), 5)

    def test_empty_store_returns_empty_subgraph(self):
        store = GraphStore()
        sg = retrieve_salient_subgraph(store)
        self.assertEqual(sg.entities, [])
        self.assertEqual(sg.facts, [])

    def test_min_confidence_filters_facts(self):
        store = GraphStore()
        store.add_entity(Entity(entity_id="e1"))
        f_high = Fact(fact_type="attribute", subject="e1", predicate="a", obj=1, confidence=0.9)
        f_low = Fact(fact_type="attribute", subject="e1", predicate="b", obj=2, confidence=0.2)
        store.add_fact(f_high)
        store.add_fact(f_low)
        config = RetrievalConfig(min_confidence=0.5)
        sg = retrieve_salient_subgraph(store, config=config)
        fact_ids = [f.fact_id for f in sg.facts]
        self.assertIn(f_high.fact_id, fact_ids)
        self.assertNotIn(f_low.fact_id, fact_ids)

    def test_scores_present_for_selected_entities(self):
        store = self._build_store()
        sg = retrieve_salient_subgraph(store)
        for entity in sg.entities:
            self.assertIn(entity.entity_id, sg.scores)

    def test_determinism(self):
        """Retrieval with the same store and config produces the same entity order."""
        store = self._build_store()
        config = RetrievalConfig()
        now_ts = time.time()
        sg1 = retrieve_salient_subgraph(store, config=config, now_ts=now_ts)
        sg2 = retrieve_salient_subgraph(store, config=config, now_ts=now_ts)
        self.assertEqual([e.entity_id for e in sg1.entities], [e.entity_id for e in sg2.entities])


# ===========================================================================
# Prompt serialiser tests
# ===========================================================================


class TestPromptSerializer(unittest.TestCase):
    def _make_subgraph(self):
        store = GraphStore()
        store.add_entity(Entity(entity_id="dog_17", entity_type="dog"))
        store.add_entity(Entity(entity_id="vehicle_12", entity_type="vehicle"))
        f_attr = Fact(fact_type="attribute", subject="dog_17", predicate="location", obj="lot_1", confidence=0.91)
        f_rel = Fact(fact_type="relation", subject="dog_17", predicate="near", obj="vehicle_12", confidence=0.84)
        store.add_fact(f_attr)
        store.add_fact(f_rel)
        return retrieve_salient_subgraph(store, mentioned_entity_ids=["dog_17", "vehicle_12"])

    def test_markers_present(self):
        sg = self._make_subgraph()
        text = serialize_subgraph(sg)
        self.assertIn(GRAPH_CONTEXT_START, text)
        self.assertIn(GRAPH_CONTEXT_END, text)

    def test_valid_json_between_markers(self):
        sg = self._make_subgraph()
        text = serialize_subgraph(sg)
        start = text.index(GRAPH_CONTEXT_START) + len(GRAPH_CONTEXT_START)
        end = text.index(GRAPH_CONTEXT_END)
        payload = json.loads(text[start:end].strip())
        self.assertIn("schema_version", payload)
        self.assertIn("entities", payload)
        self.assertIn("relation_facts", payload)
        self.assertIn("attribute_facts", payload)

    def test_schema_version_in_output(self):
        sg = self._make_subgraph()
        text = serialize_subgraph(sg)
        self.assertIn(str(SCHEMA_VERSION), text)

    def test_entity_ids_preserved(self):
        sg = self._make_subgraph()
        text = serialize_subgraph(sg)
        self.assertIn("dog_17", text)
        self.assertIn("vehicle_12", text)

    def test_scores_optional(self):
        sg = self._make_subgraph()
        without = serialize_subgraph(sg, include_scores=False)
        with_ = serialize_subgraph(sg, include_scores=True)
        self.assertNotIn("salience_score", without)
        self.assertIn("salience_score", with_)

    def test_empty_subgraph_serialises(self):
        sg = SubGraph()
        text = serialize_subgraph(sg)
        self.assertIn(GRAPH_CONTEXT_START, text)
        payload = json.loads(text.split(GRAPH_CONTEXT_START)[1].split(GRAPH_CONTEXT_END)[0].strip())
        self.assertEqual(payload["entities"], [])

    def test_update_instructions_contain_markers(self):
        instructions = build_update_instructions()
        self.assertIn(GRAPH_UPDATES_START, instructions)
        self.assertIn(GRAPH_UPDATES_END, instructions)
        self.assertIn("graph_updates", instructions)


# ===========================================================================
# Graph-update parsing tests
# ===========================================================================


class TestParseGraphUpdates(unittest.TestCase):
    def _wrap(self, payload: dict) -> str:
        return f"{GRAPH_UPDATES_START}\n{json.dumps(payload)}\n{GRAPH_UPDATES_END}"

    def test_empty_output_returns_empty_set(self):
        result = parse_graph_updates("No updates here.")
        self.assertEqual(len(result.updates), 0)

    def test_missing_start_marker_returns_empty_set(self):
        result = parse_graph_updates(f"some text\n{GRAPH_UPDATES_END}")
        self.assertEqual(len(result.updates), 0)

    def test_missing_end_marker_returns_empty_set(self):
        result = parse_graph_updates(f"{GRAPH_UPDATES_START}\n{{\"graph_updates\":[]}}")
        self.assertEqual(len(result.updates), 0)

    def test_malformed_json_returns_empty_set(self):
        raw = f"{GRAPH_UPDATES_START}\nnot valid json\n{GRAPH_UPDATES_END}"
        result = parse_graph_updates(raw)
        self.assertEqual(len(result.updates), 0)

    def test_non_dict_payload_returns_empty_set(self):
        raw = f"{GRAPH_UPDATES_START}\n[1,2,3]\n{GRAPH_UPDATES_END}"
        result = parse_graph_updates(raw)
        self.assertEqual(len(result.updates), 0)

    def test_graph_updates_not_a_list_returns_empty_set(self):
        payload = {"schema_version": 1, "graph_updates": "oops"}
        result = parse_graph_updates(self._wrap(payload))
        self.assertEqual(len(result.updates), 0)

    def test_valid_set_attribute(self):
        payload = {
            "schema_version": 1,
            "graph_updates": [
                {
                    "op": "set_attribute",
                    "entity_id": "dog_17",
                    "attribute": "location",
                    "value": "lot_1",
                    "confidence": 0.9,
                    "source": "vlm",
                }
            ],
        }
        result = parse_graph_updates(self._wrap(payload))
        self.assertEqual(len(result.updates), 1)
        u = result.updates[0]
        self.assertEqual(u.op, "set_attribute")
        self.assertEqual(u.entity_id, "dog_17")
        self.assertEqual(u.attribute, "location")
        self.assertEqual(u.value, "lot_1")
        self.assertAlmostEqual(u.confidence, 0.9)

    def test_valid_add_relation(self):
        payload = {
            "schema_version": 1,
            "graph_updates": [
                {
                    "op": "add_relation",
                    "subject": "dog_17",
                    "predicate": "near",
                    "object": "vehicle_12",
                    "confidence": 0.84,
                }
            ],
        }
        result = parse_graph_updates(self._wrap(payload))
        self.assertEqual(len(result.updates), 1)
        u = result.updates[0]
        self.assertEqual(u.op, "add_relation")
        self.assertEqual(u.subject, "dog_17")
        self.assertEqual(u.predicate, "near")
        self.assertEqual(u.object, "vehicle_12")

    def test_non_dict_update_item_skipped(self):
        payload = {
            "schema_version": 1,
            "graph_updates": ["not_a_dict", {"op": "add_entity", "entity_id": "e1"}],
        }
        result = parse_graph_updates(self._wrap(payload))
        self.assertEqual(len(result.updates), 1)
        self.assertEqual(result.updates[0].entity_id, "e1")

    def test_confidence_clamped_on_parse(self):
        payload = {
            "schema_version": 1,
            "graph_updates": [{"op": "add_entity", "entity_id": "e1", "confidence": 5.0}],
        }
        result = parse_graph_updates(self._wrap(payload))
        self.assertEqual(result.updates[0].confidence, 1.0)

    def test_output_text_before_and_after_block_preserved(self):
        """The VLM task answer must remain retrievable alongside the update block."""
        answer = "There is a brown dog near the red truck."
        updates_block = self._wrap(
            {"schema_version": 1, "graph_updates": [{"op": "add_entity", "entity_id": "e1"}]}
        )
        full_output = f"{answer}\n{updates_block}\nEnd."
        result = parse_graph_updates(full_output)
        self.assertEqual(len(result.updates), 1)
        # The task answer text is not consumed by the parser.
        self.assertIn(answer, full_output)


# ===========================================================================
# GraphUpdater reconciliation tests
# ===========================================================================


class TestGraphUpdater(unittest.TestCase):
    def setUp(self):
        self.store = GraphStore()
        self.updater = GraphUpdater(self.store, min_commit_confidence=0.5)

    def _apply(self, *updates: GraphUpdate) -> ApplyResult:
        return self.updater.apply(GraphUpdateSet(updates=list(updates)))

    # --- add_entity ---
    def test_add_entity_creates_entity(self):
        result = self._apply(GraphUpdate(op="add_entity", entity_id="dog_17", entity_type="dog", confidence=0.9))
        self.assertTrue(result.results[0].accepted)
        self.assertIsNotNone(self.store.get_entity("dog_17"))
        self.assertEqual(self.store.get_entity("dog_17").entity_type, "dog")

    def test_add_entity_idempotent(self):
        u = GraphUpdate(op="add_entity", entity_id="dog_17", entity_type="dog", confidence=1.0)
        r1 = self._apply(u)
        r2 = self._apply(u)
        self.assertTrue(r1.results[0].accepted)
        self.assertTrue(r2.results[0].accepted)
        self.assertEqual(self.store.entity_count(), 1)

    # --- set_attribute ---
    def test_set_attribute_creates_fact(self):
        self.store.add_entity(Entity(entity_id="dog_17"))
        result = self._apply(
            GraphUpdate(op="set_attribute", entity_id="dog_17", attribute="color", value="brown", confidence=0.9)
        )
        self.assertTrue(result.results[0].accepted)
        facts = self.store.facts_for_entity("dog_17")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].obj, "brown")

    def test_set_attribute_supersedes_prior_fact(self):
        self.store.add_entity(Entity(entity_id="dog_17"))
        self._apply(GraphUpdate(op="set_attribute", entity_id="dog_17", attribute="color", value="brown", confidence=0.9))
        self._apply(GraphUpdate(op="set_attribute", entity_id="dog_17", attribute="color", value="black", confidence=0.85))
        valid_facts = [f for f in self.store.facts_for_entity("dog_17", valid_only=True)]
        self.assertEqual(len(valid_facts), 1)
        self.assertEqual(valid_facts[0].obj, "black")
        # Historical fact is preserved but invalidated.
        all_facts = self.store.facts_for_entity("dog_17", valid_only=False)
        self.assertEqual(len(all_facts), 2)

    def test_set_attribute_mirrors_to_entity(self):
        self.store.add_entity(Entity(entity_id="dog_17"))
        self._apply(GraphUpdate(op="set_attribute", entity_id="dog_17", attribute="location", value="lot_1", confidence=0.9))
        entity = self.store.get_entity("dog_17")
        self.assertEqual(entity.attributes["location"], "lot_1")

    def test_set_attribute_creates_entity_if_unknown(self):
        result = self._apply(
            GraphUpdate(op="set_attribute", entity_id="new_entity", attribute="x", value=1, confidence=0.9)
        )
        self.assertTrue(result.results[0].accepted)
        self.assertTrue(self.store.has_entity("new_entity"))

    def test_set_attribute_rejected_with_require_known_entities(self):
        updater = GraphUpdater(self.store, require_known_entities=True)
        result = updater.apply(GraphUpdateSet(updates=[
            GraphUpdate(op="set_attribute", entity_id="unknown_ent", attribute="x", value=1, confidence=0.9)
        ]))
        self.assertFalse(result.results[0].accepted)

    # --- add_relation ---
    def test_add_relation_creates_fact(self):
        self.store.add_entity(Entity(entity_id="dog_17"))
        self.store.add_entity(Entity(entity_id="vehicle_12"))
        result = self._apply(
            GraphUpdate(op="add_relation", subject="dog_17", predicate="near", object="vehicle_12", confidence=0.8)
        )
        self.assertTrue(result.results[0].accepted)
        self.assertEqual(self.store.fact_count(valid_only=True), 1)

    def test_add_relation_idempotent(self):
        self.store.add_entity(Entity(entity_id="dog_17"))
        self.store.add_entity(Entity(entity_id="vehicle_12"))
        u = GraphUpdate(op="add_relation", subject="dog_17", predicate="near", object="vehicle_12", confidence=0.8)
        self._apply(u)
        self._apply(u)
        self.assertEqual(self.store.fact_count(valid_only=True), 1)

    # --- remove_relation ---
    def test_remove_relation_invalidates_fact(self):
        self.store.add_entity(Entity(entity_id="dog_17"))
        self.store.add_entity(Entity(entity_id="vehicle_12"))
        self._apply(GraphUpdate(op="add_relation", subject="dog_17", predicate="near", object="vehicle_12", confidence=0.8))
        result = self._apply(
            GraphUpdate(op="remove_relation", subject="dog_17", predicate="near", object="vehicle_12", confidence=0.9)
        )
        self.assertTrue(result.results[0].accepted)
        self.assertEqual(self.store.fact_count(valid_only=True), 0)

    def test_remove_relation_noop_when_absent(self):
        result = self._apply(
            GraphUpdate(op="remove_relation", subject="dog_17", predicate="near", object="vehicle_12", confidence=0.9)
        )
        self.assertTrue(result.results[0].accepted)
        self.assertIn("no-op", result.results[0].reason)

    # --- invalidate_fact ---
    def test_invalidate_fact_by_id(self):
        f = Fact(fact_type="attribute", subject="e1", predicate="x", obj=1)
        self.store.add_fact(f)
        result = self._apply(GraphUpdate(op="invalidate_fact", fact_id=f.fact_id, confidence=1.0))
        self.assertTrue(result.results[0].accepted)
        self.assertFalse(self.store.get_fact(f.fact_id).valid)

    def test_invalidate_unknown_fact_rejected(self):
        result = self._apply(GraphUpdate(op="invalidate_fact", fact_id="nonexistent", confidence=1.0))
        self.assertFalse(result.results[0].accepted)

    # --- low-confidence gate ---
    def test_low_confidence_update_rejected(self):
        result = self._apply(GraphUpdate(op="add_entity", entity_id="low_conf", confidence=0.3))
        self.assertFalse(result.results[0].accepted)
        self.assertFalse(self.store.has_entity("low_conf"))

    def test_exactly_threshold_confidence_accepted(self):
        result = self._apply(GraphUpdate(op="add_entity", entity_id="e1", confidence=0.5))
        self.assertTrue(result.results[0].accepted)

    # --- malformed updates ---
    def test_malformed_op_rejected(self):
        result = self._apply(GraphUpdate(op="destroy_graph", entity_id="e1", confidence=0.9))
        self.assertFalse(result.results[0].accepted)

    def test_malformed_entity_id_rejected(self):
        result = self._apply(GraphUpdate(op="add_entity", entity_id="bad id!", confidence=0.9))
        self.assertFalse(result.results[0].accepted)

    # --- empty update set ---
    def test_empty_update_set_is_noop(self):
        result = self.updater.apply(GraphUpdateSet(updates=[]))
        self.assertEqual(result.total, 0)
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.rejected, 0)

    # --- aggregate counts ---
    def test_aggregate_counts(self):
        updates = [
            GraphUpdate(op="add_entity", entity_id="e1", confidence=0.9),
            GraphUpdate(op="add_entity", entity_id="bad id!", confidence=0.9),  # rejected
            GraphUpdate(op="add_entity", entity_id="e2", confidence=0.1),       # low conf
        ]
        result = self.updater.apply(GraphUpdateSet(updates=updates))
        self.assertEqual(result.total, 3)
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.rejected, 2)


# ===========================================================================
# Full round-trip tests
# ===========================================================================


class TestRoundTrip(unittest.TestCase):
    """Demonstrate one complete round trip:
    graph -> salient context -> VLM output -> validated graph updates -> graph.
    """

    def test_round_trip(self):
        # 1. Pre-populate graph with some world state.
        store = GraphStore()
        updater = GraphUpdater(store, min_commit_confidence=0.5)

        store.add_entity(Entity(entity_id="dog_17", entity_type="dog"))
        store.add_entity(Entity(entity_id="vehicle_12", entity_type="vehicle"))

        # 2. Retrieve salient subgraph.
        sg = retrieve_salient_subgraph(
            store,
            mentioned_entity_ids=["dog_17", "vehicle_12"],
            task_keywords=["dog"],
        )
        self.assertIn("dog_17", [e.entity_id for e in sg.entities])

        # 3. Serialise context into a prompt block.
        context_block = serialize_subgraph(sg)
        self.assertIn(GRAPH_CONTEXT_START, context_block)

        # 4. Simulate a VLM response with structured graph updates.
        vlm_answer = "A brown dog is standing near a red truck."
        update_payload = {
            "schema_version": SCHEMA_VERSION,
            "graph_updates": [
                {
                    "op": "set_attribute",
                    "entity_id": "dog_17",
                    "attribute": "last_seen_location",
                    "value": "parking_lot_1",
                    "confidence": 0.91,
                    "source": "vlm",
                    "provenance": "visible in current frame",
                },
                {
                    "op": "add_relation",
                    "subject": "dog_17",
                    "predicate": "near",
                    "object": "vehicle_12",
                    "confidence": 0.84,
                    "source": "vlm",
                    "provenance": "spatial proximity in frame",
                },
            ],
        }
        vlm_output = (
            f"{vlm_answer}\n"
            f"{GRAPH_UPDATES_START}\n"
            f"{json.dumps(update_payload)}\n"
            f"{GRAPH_UPDATES_END}"
        )

        # 5. Parse proposed updates.
        update_set = parse_graph_updates(vlm_output)
        self.assertEqual(len(update_set.updates), 2)

        # 6. Validate and commit.
        apply_result = updater.apply(update_set)
        self.assertEqual(apply_result.accepted, 2)
        self.assertEqual(apply_result.rejected, 0)

        # 7. Verify graph state.
        entity = store.get_entity("dog_17")
        self.assertIsNotNone(entity)
        self.assertEqual(entity.attributes.get("last_seen_location"), "parking_lot_1")
        relation_facts = store.facts_for_relation("dog_17", "near", "vehicle_12")
        self.assertEqual(len(relation_facts), 1)
        self.assertAlmostEqual(relation_facts[0].confidence, 0.84)

        # 8. VLM task answer remains intact.
        self.assertIn(vlm_answer, vlm_output)

    def test_round_trip_with_rejected_malformed_update(self):
        """A malformed update must not affect accepted ones or the task answer."""
        store = GraphStore()
        updater = GraphUpdater(store, min_commit_confidence=0.5)
        store.add_entity(Entity(entity_id="e1"))

        vlm_output = (
            "Task answer here.\n"
            f"{GRAPH_UPDATES_START}\n"
            + json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "graph_updates": [
                        {"op": "set_attribute", "entity_id": "e1", "attribute": "state", "value": "active", "confidence": 0.9},
                        {"op": "INVALID_OP", "entity_id": "e1", "confidence": 0.9},  # malformed
                    ],
                }
            )
            + f"\n{GRAPH_UPDATES_END}"
        )
        update_set = parse_graph_updates(vlm_output)
        result = updater.apply(update_set)
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.rejected, 1)
        # Valid update was applied.
        self.assertEqual(store.get_entity("e1").attributes.get("state"), "active")

    def test_round_trip_no_updates_block(self):
        """VLM output without an update block is handled gracefully."""
        store = GraphStore()
        updater = GraphUpdater(store)
        update_set = parse_graph_updates("Just a plain task answer, no updates.")
        result = updater.apply(update_set)
        self.assertEqual(result.total, 0)

    def test_contradictory_attribute_update_replaces_prior(self):
        """Two set_attribute updates for the same entity/attribute: last one wins."""
        store = GraphStore()
        updater = GraphUpdater(store)

        update_set = GraphUpdateSet(
            updates=[
                GraphUpdate(op="set_attribute", entity_id="e1", attribute="status", value="idle", confidence=0.9),
                GraphUpdate(op="set_attribute", entity_id="e1", attribute="status", value="active", confidence=0.95),
            ]
        )
        result = updater.apply(update_set)
        self.assertEqual(result.accepted, 2)

        valid_facts = store.facts_for_entity("e1", valid_only=True)
        self.assertEqual(len(valid_facts), 1)
        self.assertEqual(valid_facts[0].obj, "active")

        # Historical fact retained.
        all_facts = store.facts_for_entity("e1", valid_only=False)
        self.assertEqual(len(all_facts), 2)

    def test_provenance_and_confidence_preserved(self):
        store = GraphStore()
        updater = GraphUpdater(store, min_commit_confidence=0.3)

        update = GraphUpdate(
            op="set_attribute",
            entity_id="e1",
            attribute="color",
            value="red",
            confidence=0.75,
            source="camera_0",
            provenance="detected in frame 42",
        )
        updater.apply(GraphUpdateSet(updates=[update]))

        facts = store.facts_for_entity("e1")
        self.assertEqual(len(facts), 1)
        self.assertAlmostEqual(facts[0].confidence, 0.75)
        self.assertEqual(facts[0].source, "camera_0")
        self.assertEqual(facts[0].provenance, "detected in frame 42")


if __name__ == "__main__":
    unittest.main()
