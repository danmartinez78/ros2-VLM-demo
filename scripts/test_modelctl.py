#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELCTL_PATH = REPO_ROOT / "scripts" / "models" / "modelctl.py"
MODELCTL_WRAPPER = REPO_ROOT / "scripts" / "modelctl"

spec = importlib.util.spec_from_file_location("edge_vlm_modelctl", MODELCTL_PATH)
modelctl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = modelctl
assert spec.loader is not None
spec.loader.exec_module(modelctl)


class ModelCtlRegistryTests(unittest.TestCase):
    def test_registry_schema_and_future_profile_shape(self) -> None:
        models, profiles, manifest = modelctl.load_registries()
        self.assertIn("cosmos-reason2-8b", models)
        self.assertIn("cosmos-reason2-2b", models)
        self.assertIn("qwen3-vl-2b-instruct", models)
        self.assertIn("thor-current", profiles)
        self.assertIn("thor-f8", profiles)
        self.assertEqual(profiles["thor-mtp-template"].decode["strategy"], "mtp")
        self.assertIn("draft", profiles["thor-mtp-template"].components)
        self.assertEqual(manifest["models"]["Cosmos-Reason2-2B"]["hf_model_id"], "nvidia/Cosmos-Reason2-2B")

    def test_unknown_model_and_profile_fail(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with self.assertRaises(modelctl.ModelCtlError):
            modelctl.resolve_model("unknown-model", models)
        with self.assertRaises(modelctl.ModelCtlError):
            modelctl.resolve_profile("unknown-profile", profiles)

    def test_model_name_normalization_and_path_safety(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        ctx = modelctl._runtime_context()
        model = modelctl.resolve_model("Cosmos_Reason2_8B", models)
        profile = modelctl.resolve_profile("thor-f8", profiles)
        paths = modelctl.engine_paths(model, profile, ctx)
        self.assertTrue(str(paths.metadata_root).startswith(str(paths.model_root)))
        self.assertEqual(paths.metadata_root, paths.model_root / "engines" / "thor-f8")

    def test_profile_specific_engine_paths_are_distinct(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        ctx = modelctl._runtime_context()
        model = models["cosmos-reason2-8b"]
        current_paths = modelctl.engine_paths(model, profiles["thor-current"], ctx)
        f8_paths = modelctl.engine_paths(model, profiles["thor-f8"], ctx)
        self.assertNotEqual(current_paths.metadata_root, f8_paths.metadata_root)
        self.assertEqual(current_paths.runtime_root, current_paths.model_root / "engine")
        self.assertEqual(f8_paths.runtime_root, f8_paths.model_root / "engines" / "thor-f8")

    def test_build_commands_include_exact_profile_limits(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-build-cmd-") as tmpdir:
            env = {"EDGE_VLM_WORKSPACE_DIR": tmpdir, "TENSORRT_EDGE_LLM_BUILD_DIR": "/opt/edge/build"}
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-f8"]
                llm_cmd = modelctl.build_llm_command(model, profile, ctx)
                visual_cmd = modelctl.build_visual_command(model, profile, ctx)
        self.assertIn("--maxBatchSize", llm_cmd)
        self.assertIn("1", llm_cmd)
        self.assertIn("--maxInputLen", llm_cmd)
        self.assertIn("2048", llm_cmd)
        self.assertIn("--maxKVCacheCapacity", llm_cmd)
        self.assertIn("4096", llm_cmd)
        self.assertIn("--maxImageTokens", visual_cmd)
        self.assertIn("2048", visual_cmd)
        self.assertIn("--maxImageTokensPerImage", visual_cmd)
        self.assertIn("512", visual_cmd)


class ModelCtlWorkspaceTests(unittest.TestCase):
    def create_prepared_artifacts(self, workspace: Path, display_name: str) -> Path:
        model_root = workspace / display_name
        quantized = model_root / "quantized"
        llm_onnx = model_root / "onnx" / "llm"
        visual_onnx = model_root / "onnx" / "visual"
        quantized.mkdir(parents=True, exist_ok=True)
        llm_onnx.mkdir(parents=True, exist_ok=True)
        visual_onnx.mkdir(parents=True, exist_ok=True)
        (quantized / "weights.safetensors").write_text("weights\n", encoding="utf-8")
        (quantized / "config.json").write_text("{}\n", encoding="utf-8")
        (llm_onnx / "model.onnx").write_text("llm\n", encoding="utf-8")
        (visual_onnx / "model.onnx").write_text("visual\n", encoding="utf-8")
        return model_root

    def create_engine_artifacts(self, model: modelctl.ModelRecord, profile: modelctl.ProfileRecord, ctx: modelctl.RuntimeContext) -> None:
        paths = modelctl.engine_paths(model, profile, ctx)
        paths.llm_dir.mkdir(parents=True, exist_ok=True)
        (paths.multimodal_dir / "visual").mkdir(parents=True, exist_ok=True)
        llm_artifacts, visual_artifacts = modelctl.required_engine_artifacts(model)
        llm_config = {
            "maxBatchSize": profile.llm["maxBatchSize"],
            "maxInputLen": profile.llm["maxInputLen"],
            "maxKVCacheCapacity": profile.llm["maxKVCacheCapacity"],
        }
        visual_config = {
            "maxImageTokens": profile.visual["maxImageTokens"],
            "maxImageTokensPerImage": profile.visual["maxImageTokensPerImage"],
        }
        for rel in llm_artifacts:
            target = paths.llm_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.name == "config.json":
                target.write_text(json.dumps(llm_config), encoding="utf-8")
            else:
                target.write_text(rel + "\n", encoding="utf-8")
        for rel in visual_artifacts:
            target = paths.multimodal_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.name == "config.json":
                target.write_text(json.dumps(visual_config), encoding="utf-8")
            else:
                target.write_text(rel + "\n", encoding="utf-8")

    def write_valid_managed_manifest(
        self, model: modelctl.ModelRecord, profile: modelctl.ProfileRecord, ctx: modelctl.RuntimeContext
    ) -> None:
        paths = modelctl.engine_paths(model, profile, ctx)
        modelctl._write_json_atomic(paths.manifest_path, modelctl._engine_manifest_payload(model, profile, ctx))

    def test_prepared_artifact_reuse_detection(self) -> None:
        models, _, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-prepared-") as tmpdir:
            workspace = Path(tmpdir)
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            with mock.patch.dict(os.environ, {"EDGE_VLM_WORKSPACE_DIR": str(workspace)}, clear=False):
                status = modelctl.prepared_status(models["cosmos-reason2-8b"], modelctl._runtime_context())
        self.assertTrue(status["prepared"])
        self.assertTrue(status["quantized"])
        self.assertTrue(status["onnx"])

    def test_engine_manifest_creation_and_validation(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-manifest-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            plugin = Path(tmpdir) / "libNvInfer_edgellm_plugin.so"
            plugin.write_text("plugin\n", encoding="utf-8")
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            env = {
                "EDGE_VLM_WORKSPACE_DIR": str(workspace),
                "EDGE_VLM_ENV_FILE": str(env_file),
                "EDGELLM_PLUGIN_PATH": str(plugin),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-f8"]
                self.create_engine_artifacts(model, profile, ctx)
                self.write_valid_managed_manifest(model, profile, ctx)
                ok, errors = modelctl.validate_engine_profile(model, profile, ctx, require_manifest=True)
        self.assertTrue(ok, errors)

    def test_manifest_mismatch_detection(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-manifest-mismatch-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env = {"EDGE_VLM_WORKSPACE_DIR": str(workspace)}
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-f8"]
                self.create_engine_artifacts(model, profile, ctx)
                paths = modelctl.engine_paths(model, profile, ctx)
                payload = modelctl._engine_manifest_payload(model, profile, ctx)
                payload["maxInputLen"] = 1024
                modelctl._write_json_atomic(paths.manifest_path, payload)
                ok, errors = modelctl.validate_engine_profile(model, profile, ctx, require_manifest=True)
        self.assertFalse(ok)
        self.assertTrue(any("maxInputLen" in error for error in errors))

    def test_activation_requires_validated_profile(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-invalid-activate-") as tmpdir:
            env = {
                "EDGE_VLM_WORKSPACE_DIR": str(Path(tmpdir) / "workspace"),
                "EDGE_VLM_ENV_FILE": str(Path(tmpdir) / "edge_vlm_env.sh"),
            }
            result = subprocess.run(
                [str(MODELCTL_WRAPPER), "activate", "cosmos-reason2-8b", "thor-f8"],
                cwd=REPO_ROOT,
                env={**os.environ, **env},
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot activate an unvalidated profile", result.stderr)

    def test_activation_writes_state_and_dynamic_env(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-activate-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            plugin = Path(tmpdir) / "libNvInfer_edgellm_plugin.so"
            plugin.write_text("plugin\n", encoding="utf-8")
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            env = {
                "EDGE_VLM_WORKSPACE_DIR": str(workspace),
                "EDGE_VLM_ENV_FILE": str(env_file),
                "EDGELLM_PLUGIN_PATH": str(plugin),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-f8"]
                self.create_engine_artifacts(model, profile, ctx)
                self.write_valid_managed_manifest(model, profile, ctx)
            result = subprocess.run(
                [str(MODELCTL_WRAPPER), "activate", "cosmos-reason2-8b", "thor-f8"],
                cwd=REPO_ROOT,
                env={**os.environ, **env},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            state_file = workspace / ".edge-vlm" / "active-profile.json"
            self.assertTrue(state_file.is_file())
            self.assertTrue(env_file.is_file())
            state = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual(state["engine_profile_id"], "thor-f8")
            source_result = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f'source {shlex.quote(str(env_file))} && printf "%s|%s|%s" "$EDGE_VLM_MODEL_ID" "$EDGE_VLM_ENGINE_PROFILE_ID" "$EDGE_VLM_LLM_ENGINE_DIR"',
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(source_result.returncode, 0, msg=source_result.stderr)
            model_id, profile_id, llm_dir = source_result.stdout.split("|")
            self.assertEqual(model_id, "cosmos-reason2-8b")
            self.assertEqual(profile_id, "thor-f8")
            self.assertEqual(llm_dir, str(workspace / "Cosmos-Reason2-8B" / "engines" / "thor-f8" / "llm"))
            self.assertFalse(any(path.name.startswith(".active-profile.json.") for path in state_file.parent.iterdir()))

    def test_legacy_engine_adoption_preserves_existing_engine(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-legacy-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            plugin = Path(tmpdir) / "libNvInfer_edgellm_plugin.so"
            plugin.write_text("plugin\n", encoding="utf-8")
            env = {
                "EDGE_VLM_WORKSPACE_DIR": str(workspace),
                "EDGE_VLM_ENV_FILE": str(env_file),
                "EDGELLM_PLUGIN_PATH": str(plugin),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-current"]
                self.create_engine_artifacts(model, profile, ctx)
                legacy_file = modelctl.engine_paths(model, profile, ctx).runtime_root / "llm" / "llm.engine"
            result = subprocess.run(
                [str(MODELCTL_WRAPPER), "activate", "Cosmos-Reason2-8B", "thor-current"],
                cwd=REPO_ROOT,
                env={**os.environ, **env},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(legacy_file.is_file())
            state = json.loads((workspace / ".edge-vlm" / "active-profile.json").read_text(encoding="utf-8"))
            self.assertEqual(state["llm_engine_dir"], str(workspace / "Cosmos-Reason2-8B" / "engine" / "llm"))
            self.assertFalse((workspace / "Cosmos-Reason2-8B" / "engines" / "thor-current" / "llm" / "llm.engine").exists())

    def test_external_layout_prepare_is_rejected_before_shell_dispatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-external-prepare-") as tmpdir:
            env = {
                **os.environ,
                "EDGE_VLM_WORKSPACE_DIR": str(Path(tmpdir) / "workspace"),
                "EDGE_VLM_ENV_FILE": str(Path(tmpdir) / "edge_vlm_env.sh"),
            }
            result = subprocess.run(
                [str(MODELCTL_WRAPPER), "prepare", "qwen3-vl-2b-instruct", "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preparation_strategy 'external_layout'", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_external_layout_build_is_rejected_before_standard_build_commands(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-external-build-") as tmpdir:
            env = {
                **os.environ,
                "EDGE_VLM_WORKSPACE_DIR": str(Path(tmpdir) / "workspace"),
                "EDGE_VLM_ENV_FILE": str(Path(tmpdir) / "edge_vlm_env.sh"),
            }
            result = subprocess.run(
                [str(MODELCTL_WRAPPER), "build", "qwen3-vl-2b-instruct", "thor-f8", "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build_strategy 'external_layout'", result.stderr)
        self.assertNotIn("llm_build", result.stdout)
        self.assertNotIn("visual_build", result.stdout)

    def test_template_profile_cannot_build_validate_or_activate(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-mtp-template-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            plugin = Path(tmpdir) / "libNvInfer_edgellm_plugin.so"
            plugin.write_text("plugin\n", encoding="utf-8")
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            env = {
                **os.environ,
                "EDGE_VLM_WORKSPACE_DIR": str(workspace),
                "EDGE_VLM_ENV_FILE": str(env_file),
                "EDGELLM_PLUGIN_PATH": str(plugin),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-mtp-template"]
                self.create_engine_artifacts(model, profile, ctx)
                self.write_valid_managed_manifest(model, profile, ctx)
                ok, errors = modelctl.validate_engine_profile(model, profile, ctx, require_manifest=True)
            self.assertFalse(ok)
            self.assertTrue(any("decode strategy 'mtp'" in error for error in errors))
            self.assertTrue(any(str(workspace / "Cosmos-Reason2-8B" / "engines" / "thor-mtp-template" / "draft") in error for error in errors))
            build = subprocess.run(
                [str(MODELCTL_WRAPPER), "build", "cosmos-reason2-8b", "thor-mtp-template", "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(build.returncode, 0)
            self.assertIn("cannot be built", build.stderr)
            activate = subprocess.run(
                [str(MODELCTL_WRAPPER), "activate", "cosmos-reason2-8b", "thor-mtp-template"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(activate.returncode, 0)
            self.assertIn("cannot be activated", activate.stderr)

    def test_activation_rollback_preserves_existing_env_when_env_write_fails(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-activate-env-fail-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            state_file = workspace / ".edge-vlm" / "active-profile.json"
            plugin = Path(tmpdir) / "libNvInfer_edgellm_plugin.so"
            plugin.write_text("plugin\n", encoding="utf-8")
            legacy_env = "#!/usr/bin/env bash\nexport EDGE_VLM_MODEL_ID=legacy\n"
            env_file.write_text(legacy_env, encoding="utf-8")
            env = {
                "EDGE_VLM_WORKSPACE_DIR": str(workspace),
                "EDGE_VLM_ENV_FILE": str(env_file),
                "EDGELLM_PLUGIN_PATH": str(plugin),
            }
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-f8"]
                self.create_engine_artifacts(model, profile, ctx)
                self.write_valid_managed_manifest(model, profile, ctx)
                with mock.patch.object(modelctl, "ensure_runtime_env_file", side_effect=OSError("env write failed")):
                    with self.assertRaises(OSError):
                        modelctl.cmd_activate(argparse.Namespace(model=model.model_id, profile=profile.profile_id, dry_run=False))
            self.assertEqual(env_file.read_text(encoding="utf-8"), legacy_env)
            self.assertFalse(state_file.exists())

    def test_activation_rollback_restores_legacy_env_on_first_migration_failure(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-activate-migration-fail-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            state_file = workspace / ".edge-vlm" / "active-profile.json"
            plugin = Path(tmpdir) / "libNvInfer_edgellm_plugin.so"
            plugin.write_text("plugin\n", encoding="utf-8")
            legacy_env = "#!/usr/bin/env bash\nexport EDGE_VLM_MODEL_ID=legacy\nexport EDGE_VLM_LLM_ENGINE_DIR=/legacy/llm\n"
            env_file.write_text(legacy_env, encoding="utf-8")
            env = {
                "EDGE_VLM_WORKSPACE_DIR": str(workspace),
                "EDGE_VLM_ENV_FILE": str(env_file),
                "EDGELLM_PLUGIN_PATH": str(plugin),
            }
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                profile = profiles["thor-f8"]
                self.create_engine_artifacts(model, profile, ctx)
                self.write_valid_managed_manifest(model, profile, ctx)
                original_write_json_atomic = modelctl._write_json_atomic

                def fail_on_active_state(path: Path, payload: dict[str, object]) -> None:
                    if path == Path(ctx.state_file):
                        raise OSError("state write failed")
                    original_write_json_atomic(path, payload)

                with mock.patch.object(modelctl, "_write_json_atomic", side_effect=fail_on_active_state):
                    with self.assertRaises(OSError):
                        modelctl.cmd_activate(argparse.Namespace(model=model.model_id, profile=profile.profile_id, dry_run=False))
            self.assertEqual(env_file.read_text(encoding="utf-8"), legacy_env)
            self.assertFalse(state_file.exists())

    def test_activation_rollback_preserves_existing_active_state_on_reactivation_failure(self) -> None:
        models, profiles, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-activate-reactivation-fail-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            plugin = Path(tmpdir) / "libNvInfer_edgellm_plugin.so"
            plugin.write_text("plugin\n", encoding="utf-8")
            env = {
                "EDGE_VLM_WORKSPACE_DIR": str(workspace),
                "EDGE_VLM_ENV_FILE": str(env_file),
                "EDGELLM_PLUGIN_PATH": str(plugin),
            }
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = modelctl._runtime_context()
                model = models["cosmos-reason2-8b"]
                current_profile = profiles["thor-current"]
                next_profile = profiles["thor-f8"]
                self.create_engine_artifacts(model, current_profile, ctx)
                self.create_engine_artifacts(model, next_profile, ctx)
                self.write_valid_managed_manifest(model, next_profile, ctx)
                current_state = modelctl._active_state_payload(model, current_profile, ctx)
                modelctl._write_json_atomic(Path(ctx.state_file), current_state)
                current_env = modelctl.render_runtime_env(ctx)
                modelctl._write_text_atomic(Path(ctx.env_file), current_env)
                original_write_json_atomic = modelctl._write_json_atomic

                def fail_on_active_state(path: Path, payload: dict[str, object]) -> None:
                    if path == Path(ctx.state_file):
                        raise OSError("state write failed")
                    original_write_json_atomic(path, payload)

                with mock.patch.object(modelctl, "_write_json_atomic", side_effect=fail_on_active_state):
                    with self.assertRaises(OSError):
                        modelctl.cmd_activate(argparse.Namespace(model=model.model_id, profile=next_profile.profile_id, dry_run=False))
            state = json.loads((workspace / ".edge-vlm" / "active-profile.json").read_text(encoding="utf-8"))
            self.assertEqual(state["engine_profile_id"], "thor-current")
            self.assertEqual(env_file.read_text(encoding="utf-8"), current_env)

    def test_cli_dry_runs_do_not_mutate_workspace(self) -> None:
        models, _, _ = modelctl.load_registries()
        with tempfile.TemporaryDirectory(prefix="edge-vlm-dry-run-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            plugin = Path(tmpdir) / "libNvInfer_edgellm_plugin.so"
            plugin.write_text("plugin\n", encoding="utf-8")
            self.create_prepared_artifacts(workspace, models["cosmos-reason2-8b"].display_name)
            env = {**os.environ, "EDGE_VLM_WORKSPACE_DIR": str(workspace), "EDGE_VLM_ENV_FILE": str(env_file), "EDGELLM_PLUGIN_PATH": str(plugin)}
            prepare = subprocess.run(
                [str(MODELCTL_WRAPPER), "prepare", "cosmos-reason2-8b", "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(prepare.returncode, 0, msg=prepare.stderr)
            self.assertIn("Prepared artifacts already available", prepare.stdout)
            build = subprocess.run(
                [str(MODELCTL_WRAPPER), "build", "cosmos-reason2-8b", "thor-f8", "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, msg=build.stderr)
            self.assertIn("--maxInputLen 2048", build.stdout)
            self.assertIn("--maxImageTokens 2048", build.stdout)
            activate = subprocess.run(
                [str(MODELCTL_WRAPPER), "activate", "cosmos-reason2-8b", "thor-f8", "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(activate.returncode, 0)
            self.assertFalse((workspace / ".edge-vlm" / "active-profile.json").exists())
            self.assertFalse(env_file.exists())

    def test_prepare_dry_run_uses_shell_flow_without_engine_build(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-prepare-cli-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            env = {**os.environ, "EDGE_VLM_WORKSPACE_DIR": str(workspace), "EDGE_VLM_ENV_FILE": str(env_file)}
            result = subprocess.run(
                [str(MODELCTL_WRAPPER), "prepare", "cosmos-reason2-2b", "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("--skip-engine-build", result.stdout)
            self.assertIn("--skip-runtime-config", result.stdout)
            self.assertNotIn("native Thor llm_build", result.stdout)
            self.assertFalse(env_file.exists())


if __name__ == "__main__":
    unittest.main()
