#!/usr/bin/env python3
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "scripts" / "thor" / "jp71_manifest.json"
SETUP_SCRIPT = REPO_ROOT / "scripts" / "prepare_thor_jp71_assets.sh"
TOP_LEVEL_SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup_thor_jp71.sh"
INSTALL_DEPENDENCIES_SCRIPT = REPO_ROOT / "scripts" / "install_dependencies.sh"
APT_GUARD_SCRIPT = REPO_ROOT / "scripts" / "apt_transaction_guard.sh"
ROS_SETUP_GUARD_SCRIPT = REPO_ROOT / "scripts" / "ros_setup_guard.sh"
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
        expected_auth_header = "Authorization: Bearer " + "$" + "{token}"
        self.assertIn(expected_auth_header, setup_script)
        self.assertIn('-H "${auth_header}"', setup_script)

    def test_engine_builders_receive_explicit_plugin_path_env(self) -> None:
        setup_script = SETUP_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('run_cmd env "EDGELLM_PLUGIN_PATH=${plugin_path}" "${llm_builder}"', setup_script)
        self.assertIn('run_cmd env "EDGELLM_PLUGIN_PATH=${plugin_path}" "${visual_builder}"', setup_script)


class RosSetupGuardTests(unittest.TestCase):
    def test_helper_sources_ros_style_script_under_nounset_and_restores_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-ros-setup-guard-") as tmpdir:
            setup_script = Path(tmpdir) / "setup.bash"
            setup_script.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        'if [ -n "$AMENT_TRACE_SETUP_FILES" ]; then',
                        "  :",
                        "fi",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "bash",
                    "-lc",
                    "\n".join(
                        [
                            "set -eu",
                            f'source "{ROS_SETUP_GUARD_SCRIPT}"',
                            "unset AMENT_TRACE_SETUP_FILES 2>/dev/null || true",
                            f'source_ros_setup_nounset_safe "{setup_script}"',
                            '[[ "$-" == *u* ]]',
                        ]
                    ),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_helper_preserves_disabled_nounset_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-ros-setup-guard-off-") as tmpdir:
            setup_script = Path(tmpdir) / "setup.bash"
            setup_script.write_text(
                'if [ -n "$AMENT_TRACE_SETUP_FILES" ]; then :; fi\n',
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "bash",
                    "-lc",
                    "\n".join(
                        [
                            "set -e",
                            f'source "{ROS_SETUP_GUARD_SCRIPT}"',
                            "unset AMENT_TRACE_SETUP_FILES 2>/dev/null || true",
                            f'source_ros_setup_nounset_safe "{setup_script}"',
                            '[[ "$-" != *u* ]]',
                        ]
                    ),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)


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
                "before the first host APT transaction, neutralize stale Isaac ROS host APT preferences from previous runs",
                result.stdout,
            )
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
            self.assertNotIn("verify Hugging Face access", result.stdout)

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
            self.assertNotIn("verify Hugging Face access", result.stdout)

    def test_dry_run_engine_plan_exports_absolute_plugin_path_outside_repo_cwd(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-plugin-env-") as tmpdir:
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
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            llm_match = re.search(
                r"EDGELLM_PLUGIN_PATH=([^\s]+)\s+native Thor llm_build",
                result.stdout,
            )
            visual_match = re.search(
                r"EDGELLM_PLUGIN_PATH=([^\s]+)\s+native Thor visual_build",
                result.stdout,
            )
            self.assertIsNotNone(llm_match)
            self.assertIsNotNone(visual_match)
            assert llm_match is not None and visual_match is not None
            self.assertTrue(Path(llm_match.group(1)).is_absolute())
            self.assertEqual(llm_match.group(1), visual_match.group(1))

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


class InstallDependenciesGuardTests(unittest.TestCase):
    def test_guard_rejects_protected_nvidia_removal(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_APT_GUARD_TEST_MODE"] = "1"
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Remv nvidia-opencv-dev [4.8.0]"

        result = subprocess.run(
            ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("removes protected package 'nvidia-opencv-dev'", result.stderr)

    def test_guard_accepts_safe_transaction(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_APT_GUARD_TEST_MODE"] = "1"
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst python3-rosdep (0.26.0)"

        result = subprocess.run(
            ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("APT guard test transaction passed.", result.stdout)

    def test_guard_rejects_arch_qualified_protected_removal(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_APT_GUARD_TEST_MODE"] = "1"
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Remv nvidia-opencv-dev:arm64 [4.8.0]"

        result = subprocess.run(
            ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("removes protected package 'nvidia-opencv-dev'", result.stderr)

    def test_guard_reports_simulation_failure(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_APT_GUARD_TEST_MODE"] = "1"
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "E: Simulated resolver failure"
        env["EDGE_VLM_APT_SIMULATION_EXIT_CODE"] = "100"

        result = subprocess.run(
            ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unable to simulate APT transaction", result.stderr)


class PrepareThorRtdetrGuardTests(unittest.TestCase):
    def test_rtdetr_guard_rejects_protected_nvidia_removal(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_APT_GUARD_TEST_MODE"] = "1"
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Remv nvidia-jetpack-dev [7.1]"

        result = subprocess.run(
            [
                str(SETUP_SCRIPT),
                "--dry-run",
                "--skip-edge-llm",
                "--skip-model",
                "--skip-data",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("removes protected package 'nvidia-jetpack-dev'", result.stderr)

    def test_rtdetr_guard_accepts_safe_transaction(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_APT_GUARD_TEST_MODE"] = "1"
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst ros-jazzy-isaac-ros-rtdetr (4.5.0)"

        result = subprocess.run(
            [
                str(SETUP_SCRIPT),
                "--dry-run",
                "--skip-edge-llm",
                "--skip-model",
                "--skip-data",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("APT guard test transaction passed for RT-DETR packages.", result.stdout)


class RosdepGuardTests(unittest.TestCase):
    def test_rosdep_guard_rejects_protected_nvidia_removal(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_ROSDEP_SIMULATION_OUTPUT"] = (
            "sudo -H apt-get install -y ros-jazzy-image-transport libopencv-dev"
        )
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Remv nvidia-jetpack [7.1-b112]"

        result = subprocess.run(
            [
                "bash",
                "-lc",
                f'source "{APT_GUARD_SCRIPT}" && assert_safe_rosdep_install_plan "{REPO_ROOT}" "jazzy"',
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("removes protected package 'nvidia-jetpack'", result.stderr)

    def test_rosdep_guard_accepts_safe_transaction(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_ROSDEP_SIMULATION_OUTPUT"] = (
            "sudo -H apt-get install -y ros-jazzy-image-transport python3-rosdep"
        )
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst python3-rosdep (0.26.0)"

        result = subprocess.run(
            [
                "bash",
                "-lc",
                f'source "{APT_GUARD_SCRIPT}" && assert_safe_rosdep_install_plan "{REPO_ROOT}" "jazzy"',
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class InstallDependenciesIsaacPreferenceGuardTests(unittest.TestCase):
    @staticmethod
    def _set_stack_policy_env(env: dict[str, str]) -> None:
        env["EDGE_VLM_CUDA_PACKAGE_FOR_TEST"] = "cuda-compiler-13-0"
        env["EDGE_VLM_APT_POLICY_LIBOPENCV_DEV_OUTPUT"] = (
            "libopencv-dev:\n"
            "  Installed: 4.8.0-3-g6ef37b4\n"
            "  Candidate: 4.8.0-3-g6ef37b4\n"
        )
        env["EDGE_VLM_APT_POLICY_LIBNVINFER_DEV_OUTPUT"] = (
            "libnvinfer-dev:\n"
            "  Installed: 10.0.1-1+cuda13.0\n"
            "  Candidate: 10.0.1-1+cuda13.0\n"
        )
        env["EDGE_VLM_APT_POLICY_CUDA_COMPILER_13_0_OUTPUT"] = (
            "cuda-compiler-13-0:\n"
            "  Installed: 13.0.0-1\n"
            "  Candidate: 13.0.0-1\n"
        )

    def test_isaac_ros_opencv_pref_is_neutralized(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-prefs-") as tmpdir:
            prefs_dir = Path(tmpdir) / "preferences.d"
            prefs_dir.mkdir(parents=True, exist_ok=True)
            pref_file = prefs_dir / "isaac-ros-opencv-4-6.pref"
            pref_file.write_text(
                "Package: libopencv*\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(prefs_dir)
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst libopencv-dev (4.8.0-3-g6ef37b4)"
            self._set_stack_policy_env(env)

            result = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertFalse(pref_file.exists())
            self.assertTrue((prefs_dir / "isaac-ros-opencv-4-6.pref.disabled-by-edge-vlm").exists())
            self.assertIn("Isaac ROS host preference guard test passed.", result.stdout)

    def test_isaac_ros_thor_stack_pref_files_are_neutralized(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-thor-prefs-") as tmpdir:
            prefs_dir = Path(tmpdir) / "preferences.d"
            prefs_dir.mkdir(parents=True, exist_ok=True)
            pref_contents = {
                "isaac-ros-opencv-4-6.pref": "Package: libopencv*\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
                "isaac-ros-cuda-13-0.pref": "Package: cuda*\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
                "isaac-ros-tensorrt-13-0.pref": "Package: libnvinfer*\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
                "isaac-ros-dgx-spark.pref": "Package: *\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
            }
            for name, content in pref_contents.items():
                (prefs_dir / name).write_text(content, encoding="utf-8")

            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(prefs_dir)
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst libopencv-dev (4.8.0-3-g6ef37b4)"
            self._set_stack_policy_env(env)

            result = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            for pref_name in pref_contents:
                self.assertFalse((prefs_dir / pref_name).exists())
                self.assertTrue((prefs_dir / f"{pref_name}.disabled-by-edge-vlm").exists())
            self.assertIn("Isaac ROS host preference guard test passed.", result.stdout)

    def test_stale_isaac_prefs_are_neutralized_before_first_simulated_transaction(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-order-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            prefs_dir = tmpdir_path / "preferences.d"
            prefs_dir.mkdir(parents=True, exist_ok=True)
            pref_contents = {
                "isaac-ros-opencv-4-6.pref": "Package: libopencv*\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
                "isaac-ros-cuda-13-0.pref": "Package: cuda*\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
                "isaac-ros-tensorrt-13-0.pref": "Package: libnvinfer*\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
                "isaac-ros-dgx-spark.pref": "Package: *\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
            }
            for name, content in pref_contents.items():
                (prefs_dir / name).write_text(content, encoding="utf-8")

            simulation_script = tmpdir_path / "simulate_apt.sh"
            simulation_script.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -eu",
                        'if compgen -G "${EDGE_VLM_APT_PREFERENCES_DIR}/isaac-ros-*.pref" >/dev/null; then',
                        '  echo "Remv nvidia-opencv-dev [7.1-b112]"',
                        "else",
                        '  echo "Inst libopencv-dev (4.8.0-3-g6ef37b4)"',
                        "fi",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            simulation_script.chmod(0o755)

            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(prefs_dir)
            env["EDGE_VLM_APT_SIMULATION_OUTPUT_COMMAND"] = str(simulation_script)
            self._set_stack_policy_env(env)

            result = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            for pref_name in pref_contents:
                self.assertFalse((prefs_dir / pref_name).exists())
                self.assertTrue((prefs_dir / f"{pref_name}.disabled-by-edge-vlm").exists())
            self.assertIn("Isaac ROS host preference guard test passed.", result.stdout)

    def test_isaac_ros_pref_guard_rejects_protected_package_removal_plan(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-prefs-removal-") as tmpdir:
            prefs_dir = Path(tmpdir) / "preferences.d"
            prefs_dir.mkdir(parents=True, exist_ok=True)
            (prefs_dir / "isaac-ros-opencv-4-6.pref").write_text(
                "Package: libopencv*\nPin: release o=Ubuntu\nPin-Priority: 1001\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(prefs_dir)
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Remv nvidia-opencv-dev [7.1-b112]"
            self._set_stack_policy_env(env)

            result = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("removes protected package 'nvidia-opencv-dev'", result.stderr)

    def test_isaac_ros_pref_guard_rejects_opencv_candidate_downgrade(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-prefs-candidate-") as tmpdir:
            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(Path(tmpdir) / "preferences.d")
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst libopencv-dev (4.8.0-3-g6ef37b4)"
            self._set_stack_policy_env(env)
            env["EDGE_VLM_APT_POLICY_LIBOPENCV_DEV_OUTPUT"] = (
                "libopencv-dev:\n"
                "  Installed: 4.8.0-3-g6ef37b4\n"
                "  Candidate: 4.6.0+dfsg-12\n"
            )

            result = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Candidate version for OpenCV development package", result.stderr)

    def test_isaac_ros_pref_guard_rejects_tensorrt_candidate_downgrade(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-prefs-trt-candidate-") as tmpdir:
            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(Path(tmpdir) / "preferences.d")
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst libopencv-dev (4.8.0-3-g6ef37b4)"
            self._set_stack_policy_env(env)
            env["EDGE_VLM_APT_POLICY_LIBNVINFER_DEV_OUTPUT"] = (
                "libnvinfer-dev:\n"
                "  Installed: 10.0.1-1+cuda13.0\n"
                "  Candidate: 9.4.0-1+cuda12.6\n"
            )

            result = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Candidate version for TensorRT development package", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
