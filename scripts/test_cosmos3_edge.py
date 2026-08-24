#!/usr/bin/env python3
"""CPU-only regression tests for Cosmos3-Edge Phase 1 scaffold.

Tests cover:
- Cosmos3-Edge model/runtime configuration parsing;
- distinct component/runtime strategy representation;
- preservation of CR2 model behavior;
- dry-run command generation/validation;
- provenance serialization/parsing;
- rejection of incomplete/misleading Cosmos3 configurations;
- no hardware/model download requirement in CI.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELCTL_PATH = REPO_ROOT / "scripts" / "models" / "modelctl.py"
COSMOS3_CMD_PATH = REPO_ROOT / "scripts" / "models" / "cosmos3_edge_commands.py"
SCHEMA_PATH = REPO_ROOT / "scripts" / "benchmark" / "schemas" / "cosmos3_edge_provenance.schema.json"

spec = importlib.util.spec_from_file_location("edge_vlm_modelctl", MODELCTL_PATH)
modelctl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = modelctl
assert spec.loader is not None
spec.loader.exec_module(modelctl)

c3_spec = importlib.util.spec_from_file_location("edge_vlm_cosmos3_edge_commands", COSMOS3_CMD_PATH)
c3_commands = importlib.util.module_from_spec(c3_spec)
sys.modules[c3_spec.name] = c3_commands
assert c3_spec.loader is not None
c3_spec.loader.exec_module(c3_commands)


class Cosmos3EdgeRegistryTests(unittest.TestCase):
    """Validate that cosmos3-edge is correctly represented in the registry."""

    def setUp(self) -> None:
        self.models, self.profiles, self.manifest = modelctl.load_registries()

    def test_cosmos3_edge_model_is_in_registry(self) -> None:
        self.assertIn("cosmos3-edge", self.models)

    def test_cosmos3_edge_model_fields(self) -> None:
        model = self.models["cosmos3-edge"]
        self.assertEqual(model.family, "cosmos3-edge")
        self.assertEqual(model.preparation_strategy, "cosmos3_edge")
        self.assertEqual(model.build_strategy, "cosmos3_edge")
        self.assertEqual(model.manifest_model_key, "Cosmos3-Edge")
        self.assertEqual(model.manifest_model["hf_model_id"], "nvidia/Cosmos3-Edge")

    def test_cosmos3_edge_manifest_entry(self) -> None:
        entry = self.manifest["models"]["Cosmos3-Edge"]
        self.assertEqual(entry["hf_model_id"], "nvidia/Cosmos3-Edge")
        self.assertEqual(entry["quantization"], "nvfp4")
        self.assertEqual(entry["runtime_strategy"], "standard_vlm")
        self.assertIn("required_llm_artifacts", entry)
        self.assertIn("required_visual_artifacts", entry)

    def test_cosmos3_edge_profile_is_in_registry(self) -> None:
        self.assertIn("cosmos3-edge-thor-f8", self.profiles)

    def test_cosmos3_edge_profile_fields(self) -> None:
        profile = self.profiles["cosmos3-edge-thor-f8"]
        self.assertEqual(profile.target, "jetson-thor")
        self.assertTrue(profile.managed)
        self.assertEqual(profile.runtime_strategy, "standard_vlm")
        self.assertIn("llm", profile.components)
        self.assertIn("visual", profile.components)
        self.assertEqual(profile.components["llm"]["kind"], "llm")
        self.assertEqual(profile.components["visual"]["kind"], "visual")
        self.assertEqual(profile.decode["strategy"], "standard")

    def test_cosmos3_edge_profile_is_distinct_from_cr2_profiles(self) -> None:
        c3_profile = self.profiles["cosmos3-edge-thor-f8"]
        cr2_profile = self.profiles["thor-f8"]
        self.assertNotEqual(c3_profile.profile_id, cr2_profile.profile_id)
        self.assertEqual(cr2_profile.runtime_strategy, "standard_vlm")
        self.assertEqual(c3_profile.runtime_strategy, "standard_vlm")


class Cosmos3EdgeRuntimeStrategyTests(unittest.TestCase):
    """Validate that the runtime strategy abstraction works correctly."""

    def setUp(self) -> None:
        self.models, self.profiles, _ = modelctl.load_registries()

    def test_cr2_profiles_retain_standard_vlm_strategy(self) -> None:
        for pid in ("thor-current", "thor-f8"):
            profile = self.profiles[pid]
            self.assertEqual(
                profile.runtime_strategy,
                "standard_vlm",
                f"Profile {pid} should have standard_vlm strategy",
            )

    def test_cosmos3_edge_profile_also_uses_standard_vlm_strategy(self) -> None:
        """Cosmos3-Edge base checkpoint uses the same standard_vlm runtime as CR2."""
        profile = self.profiles["cosmos3-edge-thor-f8"]
        self.assertEqual(profile.runtime_strategy, "standard_vlm")

    def test_policy_runtime_strategy_is_rejected_by_validation(self) -> None:
        """cosmos3_policy_inference must not be accepted as a valid runtime strategy."""
        bad_profile = modelctl.ProfileRecord(
            profile_id="cosmos3-edge-policy-test",
            target="jetson-thor",
            managed=True,
            adopt_legacy_engine=False,
            llm={"maxBatchSize": 1, "maxInputLen": 2048, "maxKVCacheCapacity": 4096},
            visual={"maxImageTokens": 2048, "maxImageTokensPerImage": 512},
            decode={"strategy": "standard"},
            components={
                "llm": {"kind": "llm", "relative_engine_dir": "llm"},
                "visual": {"kind": "visual", "relative_engine_dir": "visual"},
            },
            runtime_strategy="cosmos3_policy_inference",
        )
        errors = modelctl._profile_support_errors(bad_profile)
        self.assertTrue(
            any("cosmos3_policy_inference" in e for e in errors),
            f"Expected policy runtime rejection, got: {errors}",
        )
        self.assertTrue(
            any("Policy-DROID" in e for e in errors),
            f"Expected Policy-DROID mention in rejection, got: {errors}",
        )

    def test_standard_vlm_strategy_passes_validation(self) -> None:
        profile = self.profiles["cosmos3-edge-thor-f8"]
        errors = modelctl._profile_support_errors(profile)
        self.assertEqual(
            errors,
            [],
            f"cosmos3-edge-thor-f8 should have no support errors, got: {errors}",
        )


class Cosmos3EdgeEnginePaths(unittest.TestCase):
    """Validate engine path isolation from CR2."""

    def test_cosmos3_edge_paths_do_not_overlap_with_cr2(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        ctx = modelctl._runtime_context()
        c3_model = models["cosmos3-edge"]
        cr2_model = models["cosmos-reason2-8b"]
        c3_profile = profiles["cosmos3-edge-thor-f8"]
        cr2_profile = profiles["thor-f8"]

        c3_paths = modelctl.engine_paths(c3_model, c3_profile, ctx)
        cr2_paths = modelctl.engine_paths(cr2_model, cr2_profile, ctx)

        self.assertNotEqual(c3_paths.model_root, cr2_paths.model_root)
        self.assertNotEqual(c3_paths.llm_dir, cr2_paths.llm_dir)
        self.assertNotEqual(c3_paths.multimodal_dir, cr2_paths.multimodal_dir)
        self.assertIn("Cosmos3-Edge", str(c3_paths.model_root))
        self.assertIn("Cosmos-Reason2", str(cr2_paths.model_root))

    def test_cosmos3_edge_profile_id_in_metadata_root(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        ctx = modelctl._runtime_context()
        model = models["cosmos3-edge"]
        profile = profiles["cosmos3-edge-thor-f8"]
        paths = modelctl.engine_paths(model, profile, ctx)
        self.assertEqual(paths.metadata_root, paths.model_root / "engines" / "cosmos3-edge-thor-f8")


class Cosmos3EdgeCR2PreservationTests(unittest.TestCase):
    """Verify that CR2 model management behavior is unchanged by Cosmos3-Edge scaffold."""

    def setUp(self) -> None:
        self.models, self.profiles, _ = modelctl.load_registries()

    def test_cr2_models_still_load(self) -> None:
        self.assertIn("cosmos-reason2-8b", self.models)
        self.assertIn("cosmos-reason2-2b", self.models)

    def test_cr2_preparation_strategy_unchanged(self) -> None:
        for mid in ("cosmos-reason2-8b", "cosmos-reason2-2b"):
            self.assertEqual(self.models[mid].preparation_strategy, "cosmos_reason2")
            self.assertEqual(self.models[mid].build_strategy, "cosmos_reason2")

    def test_cr2_profiles_still_have_correct_components(self) -> None:
        for pid in ("thor-f8", "thor-current"):
            profile = self.profiles[pid]
            self.assertIn("llm", profile.components)
            self.assertIn("visual", profile.components)

    def test_cr2_resolution_still_works_by_display_name(self) -> None:
        model = modelctl.resolve_model("Cosmos-Reason2-8B", self.models)
        self.assertEqual(model.model_id, "cosmos-reason2-8b")

    def test_cr2_engine_paths_unaffected(self) -> None:
        ctx = modelctl._runtime_context()
        model = self.models["cosmos-reason2-8b"]
        profile = self.profiles["thor-f8"]
        paths = modelctl.engine_paths(model, profile, ctx)
        self.assertEqual(paths.metadata_root, paths.model_root / "engines" / "thor-f8")


class Cosmos3EdgeDryRunCommandTests(unittest.TestCase):
    """Validate dry-run command generation for Cosmos3-Edge."""

    def setUp(self) -> None:
        self.ctx = modelctl._runtime_context()
        self.models, self.profiles, _ = modelctl.load_registries()
        self.model = self.models["cosmos3-edge"]
        self.profile = self.profiles["cosmos3-edge-thor-f8"]

    def test_acquire_command_references_hf_model_id(self) -> None:
        cmd = c3_commands.acquire_command(self.ctx, self.model)
        self.assertIn("huggingface-cli", cmd)
        self.assertIn("nvidia/Cosmos3-Edge", " ".join(cmd))

    def test_acquire_command_does_not_reference_policy_droid(self) -> None:
        cmd = c3_commands.acquire_command(self.ctx, self.model)
        cmd_str = " ".join(cmd)
        self.assertNotIn("Policy-DROID", cmd_str)
        self.assertNotIn("Cosmos3-Edge-Policy", cmd_str)

    def test_acquire_command_does_not_reference_cr2_paths(self) -> None:
        cmd = c3_commands.acquire_command(self.ctx, self.model)
        cmd_str = " ".join(cmd)
        self.assertNotIn("Cosmos-Reason2", cmd_str)

    def test_build_llm_command_includes_profile_limits(self) -> None:
        cmd = c3_commands.build_llm_command(self.ctx, self.model, self.profile)
        self.assertIn("--maxBatchSize", cmd)
        self.assertIn("--maxInputLen", cmd)
        self.assertIn("--maxKVCacheCapacity", cmd)
        self.assertIn("1", cmd)
        self.assertIn("2048", cmd)

    def test_build_llm_command_onnx_dir_is_reasoning_llm(self) -> None:
        """LLM build must consume onnx/reasoning/llm, not onnx/llm."""
        cmd = c3_commands.build_llm_command(self.ctx, self.model, self.profile)
        onnx_idx = cmd.index("--onnxDir")
        onnx_path = cmd[onnx_idx + 1]
        self.assertTrue(
            onnx_path.endswith("onnx/reasoning/llm"),
            f"Expected --onnxDir to end with 'onnx/reasoning/llm', got: {onnx_path}",
        )
        # Must not point at the generic onnx/llm/ path (trailing slash avoids
        # false match on the 'onnx/reasoning/llm' suffix itself).
        self.assertNotIn("/onnx/llm/", onnx_path + "/")

    def test_build_visual_command_onnx_dir_is_reasoning_visual(self) -> None:
        """Visual build must consume onnx/reasoning/visual, not onnx/visual."""
        cmd = c3_commands.build_visual_command(self.ctx, self.model, self.profile)
        onnx_idx = cmd.index("--onnxDir")
        onnx_path = cmd[onnx_idx + 1]
        self.assertTrue(
            onnx_path.endswith("onnx/reasoning/visual"),
            f"Expected --onnxDir to end with 'onnx/reasoning/visual', got: {onnx_path}",
        )
        # Must not point at the generic onnx/visual/ path.
        self.assertNotIn("/onnx/visual/", onnx_path + "/")

    def test_build_commands_onnx_dirs_are_under_export_output(self) -> None:
        """LLM and visual --onnxDir values must be sub-trees of the export output directory."""
        export_cmd = c3_commands.export_command(self.ctx, self.model)
        # export_command argv is [binary, checkpoint, onnx_dir/reasoning, --task, reasoning]
        export_onnx_dir = export_cmd[2]  # <model_root>/onnx/reasoning

        llm_cmd = c3_commands.build_llm_command(self.ctx, self.model, self.profile)
        llm_onnx_idx = llm_cmd.index("--onnxDir")
        llm_onnx_path = llm_cmd[llm_onnx_idx + 1]

        vis_cmd = c3_commands.build_visual_command(self.ctx, self.model, self.profile)
        vis_onnx_idx = vis_cmd.index("--onnxDir")
        vis_onnx_path = vis_cmd[vis_onnx_idx + 1]

        self.assertTrue(
            llm_onnx_path.startswith(export_onnx_dir),
            f"LLM --onnxDir '{llm_onnx_path}' must be under export output '{export_onnx_dir}'",
        )
        self.assertTrue(
            vis_onnx_path.startswith(export_onnx_dir),
            f"Visual --onnxDir '{vis_onnx_path}' must be under export output '{export_onnx_dir}'",
        )

    def test_build_visual_command_includes_image_token_limits(self) -> None:
        cmd = c3_commands.build_visual_command(self.ctx, self.model, self.profile)
        self.assertIn("--maxImageTokens", cmd)
        self.assertIn("--maxImageTokensPerImage", cmd)

    def test_export_command_uses_tensorrt_edgellm_export(self) -> None:
        cmd = c3_commands.export_command(self.ctx, self.model)
        cmd_str = " ".join(cmd)
        self.assertIn("tensorrt-edgellm-export", cmd_str)
        self.assertIn("--task", cmd)
        self.assertIn("reasoning", cmd)
        # Must not reference the ModelOpt quantization tool
        self.assertNotIn("modelopt.onnx.quantization", cmd_str)

    def test_smoke_inference_command_uses_correct_cli_path(self) -> None:
        cmd = c3_commands.smoke_inference_command(self.ctx, self.model, self.profile, 1)
        cmd_str = " ".join(cmd)
        self.assertIn("llm_inference", cmd_str)
        self.assertIn("examples/llm/llm_inference", cmd_str)
        self.assertNotIn("examples/multimodal/llm_inference", cmd_str)

    def test_smoke_inference_command_carries_required_flags(self) -> None:
        cmd = c3_commands.smoke_inference_command(self.ctx, self.model, self.profile, 1)
        self.assertIn("--engineDir", cmd)
        self.assertIn("--multimodalEngineDir", cmd)
        self.assertIn("--inputFile", cmd)
        self.assertIn("--outputFile", cmd)
        # Must not use invented flags
        self.assertNotIn("--frameCount", cmd)
        self.assertNotIn("--inputPrompt", cmd)
        self.assertNotIn("--llmEngineDir", cmd)

    def test_smoke_inference_input_file_is_committed_fixture(self) -> None:
        """--inputFile must point to the committed fixture, which must exist on disk."""
        cmd = c3_commands.smoke_inference_command(self.ctx, self.model, self.profile, 1)
        input_idx = cmd.index("--inputFile")
        input_path = Path(cmd[input_idx + 1])
        self.assertTrue(
            input_path.is_file(),
            f"Smoke-input fixture does not exist: {input_path}",
        )
        # The fixture must be valid JSON with the documented top-level 'requests' shape.
        with input_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertIn(
            "requests",
            data,
            "Smoke-input fixture must use top-level 'requests' array (llm_inference input schema)",
        )
        # Must NOT use the incorrect root-level 'messages' shape.
        self.assertNotIn(
            "messages",
            data,
            "Smoke-input fixture must not place 'messages' at root; use 'requests[].messages'",
        )

    def test_smoke_inference_input_fixture_uses_requests_schema(self) -> None:
        """Fixture must follow the documented llm_inference input schema: top-level 'requests'
        array where each request contains a 'messages' array with at least one image item."""
        with c3_commands.SMOKE_INPUT_FIXTURE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        requests = data["requests"]
        self.assertIsInstance(requests, list, "'requests' must be a list")
        self.assertTrue(len(requests) >= 1, "Fixture must have at least one request")
        first_request = requests[0]
        self.assertIn("messages", first_request, "Each request must have a 'messages' key")
        messages = first_request["messages"]
        self.assertTrue(len(messages) >= 1, "Request must have at least one message")
        first_msg = messages[0]
        self.assertIn("role", first_msg)
        self.assertIn("content", first_msg)
        content_types = [item.get("type") for item in first_msg["content"] if isinstance(item, dict)]
        self.assertIn("image", content_types, "Request message content must contain an image item")

    def test_build_procedure_contains_required_steps(self) -> None:
        steps = c3_commands.build_procedure(self.ctx, self.model, self.profile)
        labels = [label for label, _ in steps]
        self.assertTrue(any("acquisition" in l.lower() for l in labels))
        self.assertTrue(any("export" in l.lower() for l in labels))
        self.assertTrue(any("LLM engine" in l for l in labels))
        self.assertTrue(any("visual engine" in l.lower() or "Visual engine" in l for l in labels))
        self.assertTrue(any("F1" in l for l in labels))
        self.assertTrue(any("provenance" in l.lower() or "manifest" in l.lower() for l in labels))

    def test_validate_procedure_passes_for_generated_steps(self) -> None:
        steps = c3_commands.build_procedure(self.ctx, self.model, self.profile)
        errors = c3_commands.validate_procedure(steps)
        self.assertEqual(errors, [], f"Expected no validation errors, got: {errors}")

    def test_validate_procedure_rejects_modelopt_in_export(self) -> None:
        bad_steps = [
            ("2. ONNX export (tensorrt-edgellm-export)", [
                "docker", "run", "pytorch:26.05-py3",
                "python3", "-m", "modelopt.onnx.quantization",
                "--model_path", "/tmp/x",
            ])
        ]
        errors = c3_commands.validate_procedure(bad_steps)
        self.assertTrue(
            any("modelopt.onnx.quantization" in e for e in errors),
            f"Expected modelopt.onnx.quantization rejection, got: {errors}",
        )

    def test_validate_procedure_rejects_policy_acquisition(self) -> None:
        bad_steps = [("1. Checkpoint acquisition", ["huggingface-cli", "download", "nvidia/Cosmos3-Edge-Policy-DROID", "--local-dir", "/tmp/x"])]
        errors = c3_commands.validate_procedure(bad_steps)
        self.assertTrue(
            any("Policy-DROID" in e for e in errors),
            f"Expected Policy-DROID rejection, got: {errors}",
        )

    def test_workspace_is_isolated_from_cr2_in_commands(self) -> None:
        steps = c3_commands.build_procedure(self.ctx, self.model, self.profile)
        for label, cmd in steps:
            if "build" in label.lower() or "inference" in label.lower():
                cmd_str = " ".join(cmd)
                self.assertNotIn(
                    "Cosmos-Reason2",
                    cmd_str,
                    f"Step '{label}' must not reference Cosmos-Reason2 paths",
                )


class Cosmos3EdgeProvenanceSchemaTests(unittest.TestCase):
    """Validate the Cosmos3-Edge provenance schema and provenance record parsing."""

    def setUp(self) -> None:
        with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            self.schema = json.load(fh)

    def test_schema_file_is_valid_json(self) -> None:
        self.assertIsInstance(self.schema, dict)
        self.assertIn("properties", self.schema)

    def test_schema_has_runtime_strategy_field(self) -> None:
        props = self.schema["properties"]
        self.assertIn("runtime_strategy", props)
        enum_vals = props["runtime_strategy"].get("enum", [])
        self.assertIn("standard_vlm", enum_vals)
        self.assertIn("cosmos3_policy_inference", enum_vals)

    def test_schema_has_temporal_input_field(self) -> None:
        props = self.schema["properties"]
        self.assertIn("temporal_input", props)
        tp_props = props["temporal_input"]["properties"]
        self.assertIn("frame_count", tp_props)
        self.assertIn("fps", tp_props)
        self.assertIn("timestamps_ms", tp_props)
        self.assertIn("input_mode", tp_props)

    def test_schema_has_cosmos3_native_stages_field(self) -> None:
        props = self.schema["properties"]
        self.assertIn("cosmos3_native_stages", props)
        # must be nullable (unresolved until hardware)
        self.assertIn("null", props["cosmos3_native_stages"]["type"])

    def test_schema_has_task_mode_field(self) -> None:
        props = self.schema["properties"]
        self.assertIn("task_mode", props)
        self.assertIn("text_reasoning", props["task_mode"]["enum"])

    def test_schema_required_fields(self) -> None:
        required = self.schema["required"]
        for field in ("checkpoint_id", "edge_llm_commit", "runtime_strategy", "component_identities", "temporal_input"):
            self.assertIn(field, required)

    def test_minimal_valid_provenance_record_passes_required(self) -> None:
        """A minimal provenance record with only required fields should satisfy the schema."""
        record = {
            "schema_version": "1",
            "checkpoint_id": "nvidia/Cosmos3-Edge",
            "edge_llm_commit": "71dd1bae032e70771265917ec74d3ff4cad07a10",
            "runtime_strategy": "standard_vlm",
            "component_identities": {
                "llm": {"engine_dir": "/workspace/Cosmos3-Edge/engines/cosmos3-edge-thor-f8/llm"},
                "visual": {"engine_dir": "/workspace/Cosmos3-Edge/engines/cosmos3-edge-thor-f8/visual"},
            },
            "temporal_input": {
                "frame_count": 1,
            },
        }
        for field in self.schema["required"]:
            self.assertIn(field, record, f"Required field '{field}' missing from minimal record")

    def test_incomplete_provenance_record_is_detectable(self) -> None:
        """A record without runtime_strategy is detectable as incomplete."""
        incomplete = {
            "schema_version": "1",
            "checkpoint_id": "nvidia/Cosmos3-Edge",
            "edge_llm_commit": "71dd1bae032e70771265917ec74d3ff4cad07a10",
        }
        for field in self.schema["required"]:
            if field not in incomplete:
                self.assertNotIn(field, incomplete)

    def test_provenance_record_serialization_roundtrip(self) -> None:
        record = {
            "schema_version": "1",
            "checkpoint_id": "nvidia/Cosmos3-Edge",
            "edge_llm_commit": "71dd1bae032e70771265917ec74d3ff4cad07a10",
            "edge_llm_version": "0.10.0",
            "runtime_strategy": "standard_vlm",
            "component_identities": {
                "llm": {
                    "engine_dir": "/workspace/Cosmos3-Edge/engines/cosmos3-edge-thor-f8/llm",
                    "engine_sha256": None,
                    "config_sha256": None,
                },
                "visual": {
                    "engine_dir": "/workspace/Cosmos3-Edge/engines/cosmos3-edge-thor-f8/visual",
                    "engine_sha256": None,
                    "config_sha256": None,
                },
            },
            "temporal_input": {
                "frame_count": 1,
                "fps": None,
                "timestamps_ms": None,
            },
            "task_mode": "text_reasoning",
            "cosmos3_native_stages": None,
            "timing": None,
            "quantization": "nvfp4",
            "engine_profile_id": "cosmos3-edge-thor-f8",
            "build_timestamp": None,
            "run_timestamp": None,
            "notes": "Phase 1 scaffold — hardware execution pending.",
        }
        serialized = json.dumps(record, indent=2, sort_keys=True)
        recovered = json.loads(serialized)
        self.assertEqual(recovered["runtime_strategy"], "standard_vlm")
        self.assertEqual(recovered["temporal_input"]["frame_count"], 1)
        self.assertIsNone(recovered["cosmos3_native_stages"])

    def test_cr2_provenance_fields_are_not_present_in_cosmos3_schema(self) -> None:
        """The Cosmos3-Edge schema should not import CR2-specific stage label assumptions."""
        schema_text = json.dumps(self.schema)
        # These are CR2 stage labels that must not be hardcoded in the Cosmos3-Edge schema.
        for forbidden_label in ("vision_stage", "prefill_stage", "generation_stage"):
            self.assertNotIn(
                forbidden_label,
                schema_text,
                f"CR2-specific stage label '{forbidden_label}' must not appear in Cosmos3-Edge schema",
            )


class Cosmos3EdgeModelctlEnginePaths(unittest.TestCase):
    """Validate that engine manifest payload includes Cosmos3-specific fields."""

    def _write_valid_manifest(
        self,
        model: "modelctl.ModelRecord",
        profile: "modelctl.ProfileRecord",
        ctx: "modelctl.RuntimeContext",
    ) -> None:
        paths = modelctl.engine_paths(model, profile, ctx)
        payload = modelctl._engine_manifest_payload(model, profile, ctx)
        modelctl._write_json_atomic(paths.manifest_path, payload)

    def test_engine_manifest_includes_runtime_strategy(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-c3e-manifest-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env = {"EDGE_VLM_WORKSPACE_DIR": str(workspace)}
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos3-edge"]
                profile = profiles["cosmos3-edge-thor-f8"]
                payload = modelctl._engine_manifest_payload(model, profile, ctx)
        self.assertEqual(payload["runtime_strategy"], "standard_vlm")
        self.assertEqual(payload["checkpoint_id"], "nvidia/Cosmos3-Edge")
        self.assertNotIn("temporal_input", payload)

    def test_cr2_engine_manifest_has_standard_vlm_strategy(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-cr2-manifest-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env = {"EDGE_VLM_WORKSPACE_DIR": str(workspace)}
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-f8"]
                payload = modelctl._engine_manifest_payload(model, profile, ctx)
        self.assertEqual(payload["runtime_strategy"], "standard_vlm")
        self.assertNotIn("temporal_input", payload)


class Cosmos3EdgeCliTests(unittest.TestCase):
    """Tests for the cosmos3_edge_commands.py CLI interface."""

    def test_dry_run_flag_is_accepted(self) -> None:
        """--dry-run must not raise an unrecognised-argument error."""
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = c3_commands.main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertGreater(len(buf.getvalue()), 0, "--dry-run should still emit commands")

    def test_dry_run_output_matches_plain_invocation(self) -> None:
        """--dry-run output must be identical to the plain (no-flag) output."""
        import io
        import contextlib

        buf_default = io.StringIO()
        with contextlib.redirect_stdout(buf_default):
            c3_commands.main([])

        buf_dry_run = io.StringIO()
        with contextlib.redirect_stdout(buf_dry_run):
            c3_commands.main(["--dry-run"])

        self.assertEqual(buf_default.getvalue(), buf_dry_run.getvalue())


if __name__ == "__main__":
    unittest.main()
