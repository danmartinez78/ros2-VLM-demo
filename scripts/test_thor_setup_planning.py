#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "scripts" / "thor" / "jp71_manifest.json"
SETUP_SCRIPT = REPO_ROOT / "scripts" / "prepare_thor_jp71_assets.sh"
TOP_LEVEL_SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup_thor_jp71.sh"
ASSETS_MANIFEST_PATH = REPO_ROOT / "scripts" / "test_data" / "manifests" / "assets_manifest.json"


class ThorSetupManifestTests(unittest.TestCase):
    def test_manifest_contains_pinned_edge_llm_commit(self) -> None:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(
            manifest["edge_llm"]["commit"],
            "7f061f21f0a581ba234a1e233c9315b89d8e47d6",
        )

    def test_manifest_declares_required_cosmos_artifacts(self) -> None:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["edge_llm"]["cuda_ctk_version"], "13.0")
        self.assertEqual(
            manifest["models"]["Cosmos-Reason2-8B"]["pytorch_container"],
            "nvcr.io/nvidia/pytorch:26.05-py3",
        )
        required_llm = set(manifest["models"]["Cosmos-Reason2-8B"]["required_llm_artifacts"])
        required_visual = set(manifest["models"]["Cosmos-Reason2-8B"]["required_visual_artifacts"])
        self.assertIn("llm.engine", required_llm)
        self.assertIn("processed_chat_template.json", required_llm)
        self.assertIn("visual/visual.engine", required_visual)

    def test_assets_manifest_covers_expected_rosbags_and_datasets(self) -> None:
        with ASSETS_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        rosbag_ids = {entry["id"] for entry in manifest["rosbags"]}
        dataset_ids = {entry["id"] for entry in manifest["datasets"]}
        self.assertEqual({"image-proc", "h264", "nvblox", "rtdetr"}, rosbag_ids)
        self.assertTrue({"jaad", "nuscenes-mini"}.issubset(dataset_ids))

    def test_hf_preflight_uses_bearer_token_header(self) -> None:
        setup_script = SETUP_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Authorization: Bearer " + "$" + "{token}", setup_script)


class ThorSetupDryRunTests(unittest.TestCase):
    def test_dry_run_generates_plan_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-env-") as tmpdir:
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            workspace = Path(tmpdir) / "workspace"
            env = os.environ.copy()
            env["EDGE_VLM_ENV_FILE"] = str(env_file)
            env["EDGE_VLM_WORKSPACE_DIR"] = str(workspace)
            result = subprocess.run(
                [
                    str(SETUP_SCRIPT),
                    "--dry-run",
                    "--skip-edge-llm",
                    "--skip-rtdetr",
                    "--skip-data",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Thor JP7.1 setup plan:", result.stdout)
            self.assertIn("planned stage: docker pull nvcr.io/nvidia/pytorch:26.05-py3", result.stdout)
            self.assertIn("tensorrt-edgellm-quantize llm", result.stdout)
            self.assertIn("tensorrt-edgellm-export", result.stdout)
            self.assertIn("native Thor llm_build", result.stdout)
            self.assertIn("native Thor visual_build", result.stdout)
            self.assertFalse(env_file.exists())

    def test_top_level_dry_run_is_non_mutating_and_skips_build_verify(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-top-level-dry-run-") as tmpdir:
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            env = os.environ.copy()
            env["EDGE_VLM_ENV_FILE"] = str(env_file)
            result = subprocess.run(
                [
                    str(TOP_LEVEL_SETUP_SCRIPT),
                    "--dry-run",
                    "--force",
                    "--skip-rosbag-download",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("DRY-RUN  install_dependencies plan:", result.stdout)
            self.assertIn(
                "Dry-run mode requested; skipping environment source, build, and verification.",
                result.stdout,
            )
            self.assertFalse(env_file.exists())

    def test_quantized_workspace_skips_quantize_stage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-quantized-only-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            quantized_dir = workspace / "Cosmos-Reason2-8B" / "quantized"
            quantized_dir.mkdir(parents=True, exist_ok=True)
            (quantized_dir / "weights.safetensors").touch()
            (quantized_dir / "config.json").touch()

            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            env = os.environ.copy()
            env["EDGE_VLM_WORKSPACE_DIR"] = str(workspace)
            env["EDGE_VLM_ENV_FILE"] = str(env_file)

            result = subprocess.run(
                [
                    str(SETUP_SCRIPT),
                    "--dry-run",
                    "--skip-edge-llm",
                    "--skip-rtdetr",
                    "--skip-data",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertNotIn("tensorrt-edgellm-quantize llm", result.stdout)
            self.assertIn("tensorrt-edgellm-export", result.stdout)
            self.assertIn("native Thor llm_build", result.stdout)
            self.assertIn("native Thor visual_build", result.stdout)

    def test_onnx_ready_workspace_skips_quantize_and_export(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-onnx-ready-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            model_root = workspace / "Cosmos-Reason2-8B"
            quantized_dir = model_root / "quantized"
            llm_onnx_dir = model_root / "onnx" / "llm"
            visual_onnx_dir = model_root / "onnx" / "visual"
            quantized_dir.mkdir(parents=True, exist_ok=True)
            llm_onnx_dir.mkdir(parents=True, exist_ok=True)
            visual_onnx_dir.mkdir(parents=True, exist_ok=True)
            (quantized_dir / "weights.safetensors").touch()
            (quantized_dir / "config.json").touch()
            (llm_onnx_dir / "llm.onnx").touch()
            (visual_onnx_dir / "visual.onnx").touch()

            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            env = os.environ.copy()
            env["EDGE_VLM_WORKSPACE_DIR"] = str(workspace)
            env["EDGE_VLM_ENV_FILE"] = str(env_file)

            result = subprocess.run(
                [
                    str(SETUP_SCRIPT),
                    "--dry-run",
                    "--skip-edge-llm",
                    "--skip-rtdetr",
                    "--skip-data",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertNotIn("tensorrt-edgellm-quantize llm", result.stdout)
            self.assertNotIn("tensorrt-edgellm-export", result.stdout)
            self.assertIn("native Thor llm_build", result.stdout)
            self.assertIn("native Thor visual_build", result.stdout)

    def test_cosmos_validation_does_not_require_qwen_workspace(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-cosmos-only-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            llm_dir = workspace / "Cosmos-Reason2-8B" / "engine" / "llm"
            visual_dir = workspace / "Cosmos-Reason2-8B" / "engine" / "visual"
            llm_dir.mkdir(parents=True, exist_ok=True)
            visual_dir.mkdir(parents=True, exist_ok=True)

            for name in (
                "llm.engine",
                "embedding.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "processed_chat_template.json",
            ):
                (llm_dir / name).touch()
            for name in ("visual.engine", "config.json", "preprocessor_config.json"):
                (visual_dir / name).touch()

            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            env = os.environ.copy()
            env["EDGE_VLM_WORKSPACE_DIR"] = str(workspace)
            env["EDGE_VLM_ENV_FILE"] = str(env_file)

            subprocess.run(
                [
                    str(SETUP_SCRIPT),
                    "--skip-edge-llm",
                    "--skip-rtdetr",
                    "--skip-data",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertTrue(env_file.exists())

    def test_missing_hf_access_fails_before_workspace_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-hf-preflight-") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            env = os.environ.copy()
            env["HOME"] = tmpdir
            env["EDGE_VLM_WORKSPACE_DIR"] = str(workspace)
            env["HUGGING_FACE_HUB_TOKEN"] = ""
            env["HF_TOKEN"] = ""

            result = subprocess.run(
                [
                    str(SETUP_SCRIPT),
                    "--skip-edge-llm",
                    "--skip-rtdetr",
                    "--skip-data",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing Hugging Face credentials", result.stderr)
            self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
