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


class ThorSetupDryRunTests(unittest.TestCase):
    def test_dry_run_generates_plan_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-env-") as tmpdir:
            env_file = Path(tmpdir) / "edge_vlm_env.sh"
            env = os.environ.copy()
            env["EDGE_VLM_ENV_FILE"] = str(env_file)
            result = subprocess.run(
                [str(SETUP_SCRIPT), "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Thor JP7.1 setup plan:", result.stdout)
            self.assertIn("DRY-RUN", result.stdout)
            self.assertFalse(env_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
