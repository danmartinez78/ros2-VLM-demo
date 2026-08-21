#!/usr/bin/env python3
import json
import os
import re
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "scripts" / "thor" / "jp72_manifest.json"
SETUP_SCRIPT = REPO_ROOT / "scripts" / "prepare_thor_jp72_assets.sh"
TOP_LEVEL_SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup_thor_jp72.sh"
INSTALL_DEPENDENCIES_SCRIPT = REPO_ROOT / "scripts" / "install_dependencies.sh"
VERIFY_DEPLOYMENT_SCRIPT = REPO_ROOT / "scripts" / "verify_deployment.sh"
APT_GUARD_SCRIPT = REPO_ROOT / "scripts" / "apt_transaction_guard.sh"
ROS_SETUP_GUARD_SCRIPT = REPO_ROOT / "scripts" / "ros_setup_guard.sh"
ASSETS_MANIFEST_PATH = REPO_ROOT / "scripts" / "test_data" / "manifests" / "assets_manifest.json"
DOWNLOAD_ROSBAGS_SCRIPT = REPO_ROOT / "scripts" / "test_data" / "download_rosbags.sh"


class ThorSetupManifestTests(unittest.TestCase):
    def test_manifest_contains_pinned_edge_llm_commit(self) -> None:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(
            manifest["edge_llm"]["commit"],
            "71dd1bae032e70771265917ec74d3ff4cad07a10",
        )
        self.assertEqual(manifest["models"]["Cosmos-Reason2-8B"]["modelopt_version"], "0.45.0")
        self.assertEqual(manifest["supported_target"]["l4t"], "# R39 (release), REVISION: 2.x")

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
            self.assertIn("Thor JP7.2 setup plan:", result.stdout)
            self.assertIn("planned stage: docker pull nvcr.io/nvidia/pytorch:26.05-py3", result.stdout)
            self.assertIn("ensure nvidia-modelopt==0.45.0", result.stdout)
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
                "before \"isaac-ros init docker\", simulate CUDA/TensorRT/NVIDIA OpenCV transactions",
                result.stdout,
            )
            self.assertIn(
                "after \"isaac-ros init docker\", verify host CUDA/TensorRT/NVIDIA OpenCV package candidates",
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


class ThorSetupDockerHandoffTests(unittest.TestCase):
    def test_top_level_setup_stops_before_prepare_when_docker_group_needs_relogin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-docker-handoff-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_bin = tmpdir_path / "bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            install_marker = tmpdir_path / "install-ran.ok"
            prepare_marker = tmpdir_path / "prepare-ran.ok"

            install_script = tmpdir_path / "install_dependencies.sh"
            install_script.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -eu",
                        f'touch "{install_marker}"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            install_script.chmod(0o755)

            prepare_script = tmpdir_path / "prepare_thor_jp72_assets.sh"
            prepare_script.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -eu",
                        f'touch "{prepare_marker}"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            prepare_script.chmod(0o755)

            (fake_bin / "docker").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
            (fake_bin / "docker").chmod(0o755)
            (fake_bin / "id").write_text("#!/usr/bin/env bash\necho 'adm sudo'\n", encoding="utf-8")
            (fake_bin / "id").chmod(0o755)
            (fake_bin / "getent").write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -eu",
                        'if [[ "${1:-}" == "group" && "${2:-}" == "docker" ]]; then',
                        '  echo "docker:x:998:${USER}"',
                        "  exit 0",
                        "fi",
                        "exit 2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (fake_bin / "getent").chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
            env["USER"] = "thoruser"
            env["EDGE_VLM_INSTALL_DEPENDENCIES_SCRIPT"] = str(install_script)
            env["EDGE_VLM_PREPARE_THOR_ASSETS_SCRIPT"] = str(prepare_script)

            result = subprocess.run(
                [str(TOP_LEVEL_SETUP_SCRIPT), "--skip-rosbag-download"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(install_marker.exists())
            self.assertFalse(prepare_marker.exists())
            self.assertIn("Docker group membership was configured", result.stderr)
            self.assertIn("Log out and back in", result.stderr)


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


class L4TGateTests(unittest.TestCase):
    @staticmethod
    def _run_l4t_gate(
        script_path: Path, release_line: str, *script_args: str
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["EDGE_VLM_L4T_GATE_TEST_MODE"] = "1"
        env["EDGE_VLM_L4T_GATE_RELEASE"] = release_line
        return subprocess.run(
            ["bash", str(script_path), *script_args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_install_gate_accepts_r39_2_1(self) -> None:
        result = self._run_l4t_gate(
            INSTALL_DEPENDENCIES_SCRIPT,
            "# R39 (release), REVISION: 2.1, GCID: 46758480, BOARD: generic, EABI: aarch64",
            "--force",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("accepted release", result.stdout)

    def test_verify_gate_accepts_r39_2_1(self) -> None:
        result = self._run_l4t_gate(
            VERIFY_DEPLOYMENT_SCRIPT,
            "# R39 (release), REVISION: 2.1, GCID: 46758480, BOARD: generic, EABI: aarch64",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("accepted release", result.stdout)

    def test_install_gate_rejects_non_2_x_revision(self) -> None:
        result = self._run_l4t_gate(
            INSTALL_DEPENDENCIES_SCRIPT,
            "# R39 (release), REVISION: 3.0, GCID: 46758480, BOARD: generic, EABI: aarch64",
            "--force",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rejected release", result.stderr)


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
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst ros-jazzy-isaac-ros-rtdetr (4.6.0)"

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

    def test_rtdetr_install_sets_deterministic_isaac_ros_ws_and_reruns_safely(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-rtdetr-isaac-ws-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_bin = tmpdir_path / "bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            marker_file = tmpdir_path / "rtdetr.ok"
            env_file = tmpdir_path / "edge_vlm_env.sh"
            ros_setup = tmpdir_path / "setup.bash"
            ros_setup.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            ros_setup.chmod(0o755)

            sudo_script = fake_bin / "sudo"
            sudo_script.write_text("#!/usr/bin/env bash\nexec \"$@\"\n", encoding="utf-8")
            sudo_script.chmod(0o755)

            apt_get_script = fake_bin / "apt-get"
            apt_get_script.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -eu",
                        'if [[ \"${1:-}\" == \"-s\" ]]; then',
                        '  echo \"Inst ros-jazzy-isaac-ros-rtdetr (4.6.0)\"',
                        "fi",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            apt_get_script.chmod(0o755)

            ros2_log = tmpdir_path / "ros2-invocations.log"
            ros2_script = fake_bin / "ros2"
            ros2_script.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -eu",
                        '[[ -n \"${ISAAC_ROS_WS:-}\" ]] || { echo \"ERROR: ISAAC_ROS_WS is not set.\" >&2; exit 1; }',
                        '[[ \"${ISAAC_ROS_WS}\" == /* ]] || { echo \"ERROR: ISAAC_ROS_WS is not absolute.\" >&2; exit 1; }',
                        'printf \"%s\\n\" \"${ISAAC_ROS_WS}\" >>\"${EDGE_VLM_TEST_ROS2_LOG}\"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ros2_script.chmod(0o755)

            ros_workspace = tmpdir_path / "thor-ros-ws"
            expected_isaac_ws = str(ros_workspace)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
            env["ROS_WORKSPACE"] = expected_isaac_ws
            env.pop("ISAAC_ROS_WS", None)
            env["EDGE_VLM_ROS_SETUP_PATH"] = str(ros_setup)
            env["EDGE_VLM_RTDETR_MARKER_FILE"] = str(marker_file)
            env["EDGE_VLM_ENV_FILE"] = str(env_file)
            env["EDGE_VLM_TEST_ROS2_LOG"] = str(ros2_log)

            first = subprocess.run(
                [
                    str(SETUP_SCRIPT),
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
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            self.assertTrue((ros_workspace / "src").is_dir())
            self.assertTrue(marker_file.exists())
            self.assertTrue(env_file.exists())
            self.assertIn(f'export ISAAC_ROS_WS="{expected_isaac_ws}"', env_file.read_text(encoding="utf-8"))
            self.assertEqual(ros2_log.read_text(encoding="utf-8").strip(), expected_isaac_ws)

            second = subprocess.run(
                [
                    str(SETUP_SCRIPT),
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
            self.assertEqual(second.returncode, 0, msg=second.stderr)
            self.assertIn("RT-DETR models installer already completed", second.stdout)
            self.assertEqual(ros2_log.read_text(encoding="utf-8").splitlines(), [expected_isaac_ws])


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
        env["EDGE_VLM_APT_POLICY_NVIDIA_OPENCV_DEV_OUTPUT"] = (
            "nvidia-opencv-dev:\n"
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

    def test_isaac_ros_pref_guard_keeps_supported_pref_files_when_stack_is_safe(self) -> None:
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
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst nvidia-opencv-dev (4.8.0-3-g6ef37b4)"
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
            self.assertTrue(pref_file.exists())
            self.assertIn("Isaac ROS host preference guard test passed.", result.stdout)

    def test_isaac_ros_pref_guard_keeps_thor_pref_set_when_stack_is_safe(self) -> None:
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
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst nvidia-opencv-dev (4.8.0-3-g6ef37b4)"
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
                self.assertTrue((prefs_dir / pref_name).exists())
            self.assertIn("Isaac ROS host preference guard test passed.", result.stdout)

    def test_stale_isaac_pref_set_fails_if_simulated_plan_is_unsafe(self) -> None:
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
                        '  echo "Inst nvidia-opencv-dev (4.8.0-3-g6ef37b4)"',
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
            self.assertNotEqual(result.returncode, 0)
            for pref_name in pref_contents:
                self.assertTrue((prefs_dir / pref_name).exists())
            self.assertIn("removes protected package 'nvidia-opencv-dev'", result.stderr)

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
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst nvidia-opencv-dev (4.8.0-3-g6ef37b4)"
            self._set_stack_policy_env(env)
            env["EDGE_VLM_APT_POLICY_NVIDIA_OPENCV_DEV_OUTPUT"] = (
                "nvidia-opencv-dev:\n"
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
            self.assertIn("Candidate version for NVIDIA OpenCV development package", result.stderr)

    def test_isaac_ros_pref_guard_rejects_tensorrt_candidate_downgrade(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-prefs-trt-candidate-") as tmpdir:
            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(Path(tmpdir) / "preferences.d")
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst nvidia-opencv-dev (4.8.0-3-g6ef37b4)"
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

    def test_isaac_ros_pref_guard_neutralizes_incompatible_opencv_pin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-prefs-neutralize-") as tmpdir:
            prefs_dir = Path(tmpdir) / "preferences.d"
            prefs_dir.mkdir(parents=True, exist_ok=True)
            opencv_pref = prefs_dir / "isaac-ros-opencv-4-6.pref"
            trt_pref = prefs_dir / "isaac-ros-tensorrt-13-0.pref"
            opencv_pref.write_text(
                "Package: libopencv*\nPin: version 4.6.0*\nPin-Priority: 1001\n",
                encoding="utf-8",
            )
            trt_pref.write_text(
                "Package: libnvinfer*\nPin: version 10.16.*\nPin-Priority: 1001\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(prefs_dir)
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst nvidia-opencv-dev (7.2.1-b49)"
            self._set_stack_policy_env(env)
            env["EDGE_VLM_APT_POLICY_LIBOPENCV_DEV_OUTPUT"] = (
                "libopencv-dev:\n"
                "  Installed: 4.8.0-4-g18251aa\n"
                "  Candidate: 4.6.0+dfsg-13.1ubuntu1\n"
            )

            result = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertFalse(opencv_pref.exists())
            self.assertTrue((prefs_dir / "isaac-ros-opencv-4-6.pref.edge-vlm-disabled").exists())
            self.assertTrue(trt_pref.exists())
            self.assertIn("Neutralized incompatible Isaac ROS host OpenCV pin", result.stdout)

    def test_isaac_ros_pref_guard_neutralization_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-isaac-prefs-idempotent-") as tmpdir:
            prefs_dir = Path(tmpdir) / "preferences.d"
            prefs_dir.mkdir(parents=True, exist_ok=True)
            disabled_pref = prefs_dir / "isaac-ros-opencv-4-6.pref.edge-vlm-disabled"
            disabled_pref.write_text(
                "Package: libopencv*\nPin: version 4.6.0*\nPin-Priority: 1001\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
            env["EDGE_VLM_APT_PREFERENCES_DIR"] = str(prefs_dir)
            env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst nvidia-opencv-dev (7.2.1-b49)"
            self._set_stack_policy_env(env)
            env["EDGE_VLM_APT_POLICY_LIBOPENCV_DEV_OUTPUT"] = (
                "libopencv-dev:\n"
                "  Installed: 4.8.0-4-g18251aa\n"
                "  Candidate: 4.8.0-4-g18251aa\n"
            )

            first = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            second = subprocess.run(
                ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(first.returncode, 0, msg=first.stderr)
            self.assertEqual(second.returncode, 0, msg=second.stderr)
            self.assertTrue(disabled_pref.exists())
            self.assertNotIn("Neutralized incompatible Isaac ROS host OpenCV pin", second.stdout)

    def test_isaac_ros_pref_guard_pre_jetpack_allows_missing_opencv_stack(self) -> None:
        env = os.environ.copy()
        env["EDGE_VLM_ISAAC_PREF_GUARD_TEST_MODE"] = "1"
        env["EDGE_VLM_ISAAC_PREF_GUARD_PHASE"] = "pre-jetpack-install"
        env["EDGE_VLM_APT_SIMULATION_OUTPUT"] = "Inst nvidia-opencv-dev (4.8.0-3-g6ef37b4)"
        env["EDGE_VLM_APT_POLICY_NVIDIA_OPENCV_DEV_OUTPUT"] = (
            "nvidia-opencv-dev:\n"
            "  Installed: (none)\n"
            "  Candidate: 4.8.0-3-g6ef37b4\n"
        )
        env["EDGE_VLM_APT_POLICY_LIBNVINFER_DEV_OUTPUT"] = (
            "libnvinfer-dev:\n"
            "  Installed: (none)\n"
            "  Candidate: 10.0.1-1+cuda13.0\n"
        )

        result = subprocess.run(
            ["bash", str(INSTALL_DEPENDENCIES_SCRIPT), "--force"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Isaac ROS host preference guard test passed.", result.stdout)


class DownloadRosbagsArchiveFormatTests(unittest.TestCase):
    @staticmethod
    def _write_fake_curl(fake_curl_path: Path) -> None:
        fake_curl_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -eu",
                    'url="${@: -1}"',
                    'if [[ "${url}" == */files/quickstart.tar.gz ]]; then',
                    '  out_file=""',
                    "  while [[ $# -gt 0 ]]; do",
                    '    if [[ "$1" == "-o" ]]; then',
                    '      out_file="$2"',
                    "      shift 2",
                    "      continue",
                    "    fi",
                    "    shift",
                    "  done",
                    '  [[ -n "${out_file}" ]]',
                    '  cp "${EDGE_VLM_TEST_ARCHIVE_PATH}" "${out_file}"',
                    "  exit 0",
                    "fi",
                    'if [[ "${url}" == */versions ]]; then',
                    '  printf \'%s\\n\' \'{"recipeVersions":[{"versionId":"4.0.0"},{"versionId":"4.6.0"},{"versionId":"4.7.0"}]}\'',
                    "  exit 0",
                    "fi",
                    'echo "Unexpected curl URL: ${url}" >&2',
                    "exit 2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        fake_curl_path.chmod(0o755)

    @staticmethod
    def _create_sample_archive(archive_path: Path, compressed: bool) -> None:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-rosbag-archive-src-") as source_dir:
            source_path = Path(source_dir)
            bag_root = source_path / "isaac_ros_nvblox" / "quickstart"
            bag_root.mkdir(parents=True, exist_ok=True)
            (bag_root / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n", encoding="utf-8")
            mode = "w:gz" if compressed else "w"
            with tarfile.open(archive_path, mode) as archive:
                archive.add(source_path / "isaac_ros_nvblox", arcname="isaac_ros_nvblox")

    def _run_download_with_archive(self, *, compressed: bool) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="edge-vlm-rosbag-download-test-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_bin = tmpdir_path / "bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            archive_path = tmpdir_path / "quickstart.tar.gz"
            self._create_sample_archive(archive_path, compressed=compressed)
            self._write_fake_curl(fake_bin / "curl")

            output_root = tmpdir_path / "rosbags"
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
            env["ROSBAG_DIR"] = str(output_root)
            env["EDGE_VLM_TEST_ARCHIVE_PATH"] = str(archive_path)

            first = subprocess.run(
                ["bash", str(DOWNLOAD_ROSBAGS_SCRIPT), "download", "nvblox"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            target = output_root / "nvblox"
            self.assertTrue((target / ".ngc-version").exists())
            self.assertEqual((target / ".ngc-version").read_text(encoding="utf-8").strip(), "4.6.0")
            self.assertTrue((target / "isaac_ros_nvblox" / "quickstart" / "metadata.yaml").exists())

            second = subprocess.run(
                ["bash", str(DOWNLOAD_ROSBAGS_SCRIPT), "download", "nvblox"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(second.returncode, 0, msg=second.stderr)
            self.assertIn("already installed", second.stdout)
            return second

    def test_download_accepts_gzip_tar_archive(self) -> None:
        self._run_download_with_archive(compressed=True)

    def test_download_accepts_plain_tar_with_tar_gz_suffix(self) -> None:
        self._run_download_with_archive(compressed=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
