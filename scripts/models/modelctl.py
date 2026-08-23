#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


THIS_FILE = Path(__file__).resolve()
REGISTRY_DIR = THIS_FILE.parent
SCRIPTS_DIR = REGISTRY_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
MODELS_REGISTRY_PATH = REGISTRY_DIR / "models.json"
PROFILES_REGISTRY_PATH = REGISTRY_DIR / "engine_profiles.json"
THOR_MANIFEST_PATH = SCRIPTS_DIR / "thor" / "jp72_manifest.json"
PREPARE_SCRIPT_PATH = SCRIPTS_DIR / "prepare_thor_jp72_assets.sh"
VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class ModelCtlError(RuntimeError):
    """Raised for expected command failures."""


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    display_name: str
    manifest_model_key: str
    family: str
    preparation_strategy: str
    build_strategy: str
    manifest_model: dict[str, Any]


@dataclass(frozen=True)
class ProfileRecord:
    profile_id: str
    target: str
    managed: bool
    adopt_legacy_engine: bool
    llm: dict[str, Any]
    visual: dict[str, Any]
    decode: dict[str, Any]
    components: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class EnginePaths:
    model_root: Path
    metadata_root: Path
    runtime_root: Path
    llm_dir: Path
    multimodal_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class RuntimeContext:
    ros_distro: str
    ros_workspace: str
    isaac_ros_ws: str
    edge_root: str
    edge_build: str
    workspace_dir: str
    plugin_path: str
    env_file: str
    state_file: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_lookup(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise ModelCtlError("Identifier must not be empty.")
    return normalized


def _validate_registry_id(value: str, kind: str) -> str:
    if not VALID_ID_RE.fullmatch(value):
        raise ModelCtlError(f"Invalid {kind} id '{value}'. Use lowercase letters, digits, and hyphens only.")
    return value


def _expand_path(value: str) -> str:
    return os.path.abspath(os.path.expanduser(value.replace("${HOME}", str(Path.home()))))


def _infer_ros_workspace() -> str:
    env_value = os.environ.get("ROS_WORKSPACE")
    if env_value:
        return _expand_path(env_value)
    if REPO_ROOT.parent.name == "src":
        return str(REPO_ROOT.parent.parent.resolve())
    return str((Path.home() / "ros2_ws").resolve())


def _runtime_context() -> RuntimeContext:
    manifest = _load_json(THOR_MANIFEST_PATH)
    edge_root = _expand_path(os.environ.get("TENSORRT_EDGE_LLM_ROOT", manifest["edge_llm"]["default_root"]))
    edge_build = _expand_path(os.environ.get("TENSORRT_EDGE_LLM_BUILD_DIR", manifest["edge_llm"]["default_build_dir"]))
    workspace_dir = _expand_path(os.environ.get("EDGE_VLM_WORKSPACE_DIR", manifest["default_workspace"]))
    ros_workspace = _infer_ros_workspace()
    isaac_ros_ws = _expand_path(os.environ.get("ISAAC_ROS_WS", ros_workspace))
    env_file = _expand_path(os.environ.get("EDGE_VLM_ENV_FILE", str(SCRIPTS_DIR / "edge_vlm_env.sh")))
    plugin_path = _expand_path(
        os.environ.get("EDGELLM_PLUGIN_PATH", str(Path(edge_build) / manifest["edge_llm"]["plugin_name"]))
    )
    state_file = os.path.join(workspace_dir, ".edge-vlm", "active-profile.json")
    return RuntimeContext(
        ros_distro=os.environ.get("ROS_DISTRO", "jazzy"),
        ros_workspace=ros_workspace,
        isaac_ros_ws=isaac_ros_ws,
        edge_root=edge_root,
        edge_build=edge_build,
        workspace_dir=workspace_dir,
        plugin_path=plugin_path,
        env_file=env_file,
        state_file=state_file,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _write_text_atomic(path: Path, content: str, mode: int = 0o755) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def load_registries() -> tuple[dict[str, ModelRecord], dict[str, ProfileRecord], dict[str, Any]]:
    manifest = _load_json(THOR_MANIFEST_PATH)
    models_data = _load_json(MODELS_REGISTRY_PATH)
    profiles_data = _load_json(PROFILES_REGISTRY_PATH)
    if models_data.get("schema_version") != 1:
        raise ModelCtlError("Unsupported models.json schema_version.")
    if profiles_data.get("schema_version") != 1:
        raise ModelCtlError("Unsupported engine_profiles.json schema_version.")

    models: dict[str, ModelRecord] = {}
    seen_lookup: dict[str, str] = {}
    for model_id, data in models_data.get("models", {}).items():
        _validate_registry_id(model_id, "model")
        manifest_key = data["manifest_model_key"]
        manifest_model = manifest.get("models", {}).get(manifest_key)
        if manifest_model is None:
            raise ModelCtlError(f"Manifest model '{manifest_key}' referenced by {model_id} is missing.")
        record = ModelRecord(
            model_id=model_id,
            display_name=data["display_name"],
            manifest_model_key=manifest_key,
            family=data["family"],
            preparation_strategy=data["preparation_strategy"],
            build_strategy=data["build_strategy"],
            manifest_model=manifest_model,
        )
        models[model_id] = record
        for candidate in (model_id, record.display_name, manifest_key):
            lookup = _normalize_lookup(candidate)
            owner = seen_lookup.get(lookup)
            if owner is not None and owner != model_id:
                raise ModelCtlError(
                    f"Model registry lookup alias collision: '{candidate}' normalizes to '{lookup}', "
                    f"already owned by '{owner}'."
                )
            seen_lookup[lookup] = model_id

    profiles: dict[str, ProfileRecord] = {}
    profile_roots: set[str] = set()
    for profile_id, data in profiles_data.get("profiles", {}).items():
        _validate_registry_id(profile_id, "profile")
        components = data.get("components", {})
        if not components:
            raise ModelCtlError(f"Profile {profile_id} must define at least one component.")
        runtime_dirs = [str(component["relative_engine_dir"]) for component in components.values()]
        if len(runtime_dirs) != len(set(runtime_dirs)):
            raise ModelCtlError(f"Profile {profile_id} reuses the same component directory more than once.")
        if profile_id in profile_roots:
            raise ModelCtlError(f"Profile {profile_id} resolves to a colliding engine path.")
        profile_roots.add(profile_id)
        profiles[profile_id] = ProfileRecord(
            profile_id=profile_id,
            target=data["target"],
            managed=bool(data.get("managed", True)),
            adopt_legacy_engine=bool(data.get("adopt_legacy_engine", False)),
            llm=data["llm"],
            visual=data["visual"],
            decode=data["decode"],
            components=components,
        )
    return models, profiles, manifest


def resolve_model(model_ref: str, models: dict[str, ModelRecord]) -> ModelRecord:
    lookup = _normalize_lookup(model_ref)
    for record in models.values():
        if lookup in {
            record.model_id,
            _normalize_lookup(record.display_name),
            _normalize_lookup(record.manifest_model_key),
        }:
            return record
    raise ModelCtlError(f"Unknown model '{model_ref}'.")


def resolve_profile(profile_ref: str, profiles: dict[str, ProfileRecord]) -> ProfileRecord:
    lookup = _normalize_lookup(profile_ref)
    for record in profiles.values():
        if lookup == record.profile_id:
            return record
    raise ModelCtlError(f"Unknown profile '{profile_ref}'.")


def model_root(model: ModelRecord, ctx: RuntimeContext) -> Path:
    return Path(ctx.workspace_dir) / model.display_name


def prepared_status(model: ModelRecord, ctx: RuntimeContext) -> dict[str, bool]:
    root = model_root(model, ctx)
    quantized_dir = root / "quantized"
    onnx_llm_dir = root / "onnx" / "llm"
    onnx_visual_dir = root / "onnx" / "visual"
    quantized_ready = quantized_dir.is_dir() and any(
        path.is_file() and path.suffix in {".json", ".safetensors"}
        for path in quantized_dir.rglob("*")
    )
    onnx_ready = (
        onnx_llm_dir.is_dir()
        and onnx_visual_dir.is_dir()
        and any(path.is_file() and path.suffix == ".onnx" for path in onnx_llm_dir.rglob("*"))
        and any(path.is_file() and path.suffix == ".onnx" for path in onnx_visual_dir.rglob("*"))
    )
    return {
        "checkpoint": bool(model.manifest_model.get("hf_model_id")),
        "quantized": quantized_ready,
        "onnx": onnx_ready,
        "prepared": quantized_ready and onnx_ready,
    }


def engine_paths(model: ModelRecord, profile: ProfileRecord, ctx: RuntimeContext) -> EnginePaths:
    root = model_root(model, ctx)
    metadata_root = root / "engines" / profile.profile_id
    if profile.adopt_legacy_engine:
        runtime_root = root / "engine"
    else:
        runtime_root = metadata_root
    llm_dir = runtime_root / profile.components["llm"]["relative_engine_dir"]
    return EnginePaths(
        model_root=root,
        metadata_root=metadata_root,
        runtime_root=runtime_root,
        llm_dir=llm_dir,
        multimodal_dir=runtime_root,
        manifest_path=metadata_root / "engine-manifest.json",
    )


def required_engine_artifacts(model: ModelRecord) -> tuple[list[str], list[str]]:
    llm = list(model.manifest_model.get("required_llm_artifacts", []))
    visual = list(model.manifest_model.get("required_visual_artifacts", []))
    return llm, visual


def check_engine_artifacts(model: ModelRecord, paths: EnginePaths) -> list[str]:
    missing: list[str] = []
    if not paths.runtime_root.is_dir():
        missing.append(str(paths.runtime_root))
        return missing
    llm_artifacts, visual_artifacts = required_engine_artifacts(model)
    required_directories = list(model.manifest_model.get("required_directories", []))
    for rel_dir in required_directories:
        candidate = paths.model_root / rel_dir
        if not candidate.is_dir():
            missing.append(str(candidate))
    for rel in llm_artifacts:
        candidate = paths.llm_dir / rel
        if not candidate.is_file():
            missing.append(str(candidate))
    for rel in visual_artifacts:
        candidate = paths.multimodal_dir / rel
        if not candidate.is_file():
            missing.append(str(candidate))
    return missing


def _find_first_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        for payload_key, value in payload.items():
            if payload_key == key:
                return value
            found = _find_first_value(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first_value(item, key)
            if found is not None:
                return found
    return None


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_inventory_digest(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"path": str(root), "exists": False}
    digest = hashlib.sha256()
    file_count = 0
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        rel = file_path.relative_to(root).as_posix()
        stat = file_path.stat()
        digest.update(rel.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        file_count += 1
    return {
        "path": str(root),
        "exists": True,
        "file_count": file_count,
        "inventory_sha256": digest.hexdigest(),
    }


def _load_manifest_data(paths: EnginePaths) -> dict[str, Any] | None:
    return _load_json_if_exists(paths.manifest_path)


def _validate_limits_against_configs(paths: EnginePaths, profile: ProfileRecord) -> list[str]:
    mismatches: list[str] = []
    llm_config = _load_json_if_exists(paths.llm_dir / "config.json")
    if llm_config is not None:
        for key in ("maxBatchSize", "maxInputLen", "maxKVCacheCapacity"):
            actual = _find_first_value(llm_config, key)
            expected = profile.llm.get(key)
            if actual is not None and expected is not None and actual != expected:
                mismatches.append(f"llm config {key}={actual} does not match profile value {expected}")
    visual_config = _load_json_if_exists(paths.multimodal_dir / "visual" / "config.json")
    if visual_config is not None:
        for key in ("maxImageTokens", "maxImageTokensPerImage"):
            actual = _find_first_value(visual_config, key)
            expected = profile.visual.get(key)
            if actual is not None and expected is not None and actual != expected:
                mismatches.append(f"visual config {key}={actual} does not match profile value {expected}")
    return mismatches


def validate_engine_profile(
    model: ModelRecord,
    profile: ProfileRecord,
    ctx: RuntimeContext,
    *,
    require_manifest: bool,
) -> tuple[bool, list[str]]:
    paths = engine_paths(model, profile, ctx)
    errors = check_engine_artifacts(model, paths)
    errors.extend(_validate_limits_against_configs(paths, profile))
    manifest_data = _load_manifest_data(paths)
    if require_manifest:
        if manifest_data is None:
            errors.append(f"Missing engine manifest: {paths.manifest_path}")
        else:
            expected_pairs = {
                "schema_version": 1,
                "model_id": model.model_id,
                "checkpoint_id": model.manifest_model.get("hf_model_id"),
                "quantization": model.manifest_model.get("quantization"),
                "engine_profile_id": profile.profile_id,
                "target": profile.target,
                "maxBatchSize": profile.llm.get("maxBatchSize"),
                "maxInputLen": profile.llm.get("maxInputLen"),
                "maxKVCacheCapacity": profile.llm.get("maxKVCacheCapacity"),
                "maxImageTokens": profile.visual.get("maxImageTokens"),
                "maxImageTokensPerImage": profile.visual.get("maxImageTokensPerImage"),
            }
            for key, expected in expected_pairs.items():
                if manifest_data.get(key) != expected:
                    errors.append(
                        f"Engine manifest mismatch for {key}: expected {expected!r}, found {manifest_data.get(key)!r}"
                    )
            if manifest_data.get("decode") != profile.decode:
                errors.append("Engine manifest decode settings do not match registry profile.")
            prepared = manifest_data.get("prepared_artifacts", {})
            quantized_digest = _collect_inventory_digest(paths.model_root / "quantized")
            onnx_digest = _collect_inventory_digest(paths.model_root / "onnx")
            if prepared.get("quantized", {}).get("inventory_sha256") != quantized_digest.get("inventory_sha256"):
                errors.append("Prepared quantized artifact digest does not match engine manifest.")
            if prepared.get("onnx", {}).get("inventory_sha256") != onnx_digest.get("inventory_sha256"):
                errors.append("Prepared ONNX artifact digest does not match engine manifest.")
    return (not errors, errors)


def _engine_manifest_payload(model: ModelRecord, profile: ProfileRecord, ctx: RuntimeContext) -> dict[str, Any]:
    paths = engine_paths(model, profile, ctx)
    manifest = _load_json(THOR_MANIFEST_PATH)
    return {
        "schema_version": 1,
        "build_timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_id": model.model_id,
        "model_name": model.display_name,
        "checkpoint_id": model.manifest_model.get("hf_model_id"),
        "quantization": model.manifest_model.get("quantization"),
        "engine_profile_id": profile.profile_id,
        "target": profile.target,
        "tensorrt_edge_llm_commit": manifest["edge_llm"]["commit"],
        "maxBatchSize": profile.llm.get("maxBatchSize"),
        "maxInputLen": profile.llm.get("maxInputLen"),
        "maxKVCacheCapacity": profile.llm.get("maxKVCacheCapacity"),
        "maxImageTokens": profile.visual.get("maxImageTokens"),
        "maxImageTokensPerImage": profile.visual.get("maxImageTokensPerImage"),
        "decode": profile.decode,
        "components": profile.components,
        "engine_paths": {
            "metadata_root": str(paths.metadata_root),
            "runtime_root": str(paths.runtime_root),
            "llm_dir": str(paths.llm_dir),
            "multimodal_dir": str(paths.multimodal_dir),
        },
        "prepared_artifacts": {
            "quantized": _collect_inventory_digest(paths.model_root / "quantized"),
            "onnx": _collect_inventory_digest(paths.model_root / "onnx"),
        },
    }


def _managed_engine_valid(model: ModelRecord, profile: ProfileRecord, ctx: RuntimeContext) -> bool:
    ok, _ = validate_engine_profile(model, profile, ctx, require_manifest=True)
    return ok


def render_runtime_env(ctx: RuntimeContext, *, include_active_lookup: bool = True) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by scripts/modelctl",
        f'export ROS_DISTRO={shlex.quote(ctx.ros_distro)}',
        f'export ROS_WORKSPACE={shlex.quote(ctx.ros_workspace)}',
        f'export TENSORRT_EDGE_LLM_ROOT={shlex.quote(ctx.edge_root)}',
        f'export TENSORRT_EDGE_LLM_BUILD_DIR={shlex.quote(ctx.edge_build)}',
        f'export EDGE_VLM_WORKSPACE_DIR={shlex.quote(ctx.workspace_dir)}',
        f'export EDGELLM_PLUGIN_PATH={shlex.quote(ctx.plugin_path)}',
        f'export ISAAC_ROS_WS={shlex.quote(ctx.isaac_ros_ws)}',
        f'export EDGE_VLM_RUNTIME_STATE_FILE={shlex.quote(ctx.state_file)}',
        f'export EDGE_VLM_MODELCTL_PATH={shlex.quote(str(THIS_FILE))}',
    ]
    if include_active_lookup:
        lines.extend(
            [
                'if [[ -f "${EDGE_VLM_RUNTIME_STATE_FILE}" ]]; then',
                '  eval "$(python3 "${EDGE_VLM_MODELCTL_PATH}" print-env --state-file "${EDGE_VLM_RUNTIME_STATE_FILE}")"',
                "fi",
            ]
        )
    return "\n".join(lines) + "\n"


def ensure_runtime_env_file(ctx: RuntimeContext, *, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN  write runtime env wrapper {ctx.env_file}")
        return
    _write_text_atomic(Path(ctx.env_file), render_runtime_env(ctx))


def _active_state_payload(model: ModelRecord, profile: ProfileRecord, ctx: RuntimeContext) -> dict[str, Any]:
    paths = engine_paths(model, profile, ctx)
    return {
        "schema_version": 1,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_id": model.model_id,
        "model_name": model.display_name,
        "engine_profile_id": profile.profile_id,
        "target": profile.target,
        "workspace_dir": ctx.workspace_dir,
        "llm_engine_dir": str(paths.llm_dir),
        "multimodal_engine_dir": str(paths.multimodal_dir),
        "plugin_path": ctx.plugin_path,
        "state_file": ctx.state_file,
        "env_file": ctx.env_file,
        "manifest_path": str(paths.manifest_path),
        "decode": profile.decode,
    }


def read_active_state(state_file: Path | None = None) -> dict[str, Any]:
    path = state_file or Path(_runtime_context().state_file)
    if not path.is_file():
        raise ModelCtlError(f"No active runtime state at {path}")
    payload = _load_json(path)
    if payload.get("schema_version") != 1:
        raise ModelCtlError(f"Unsupported runtime state schema in {path}")
    return payload


def build_llm_command(model: ModelRecord, profile: ProfileRecord, ctx: RuntimeContext) -> list[str]:
    paths = engine_paths(model, profile, ctx)
    return [
        str(Path(ctx.edge_build) / "examples" / "llm" / "llm_build"),
        "--onnxDir",
        str(paths.model_root / "onnx" / "llm"),
        "--engineDir",
        str(paths.llm_dir),
        "--maxBatchSize",
        str(profile.llm["maxBatchSize"]),
        "--maxInputLen",
        str(profile.llm["maxInputLen"]),
        "--maxKVCacheCapacity",
        str(profile.llm["maxKVCacheCapacity"]),
    ]


def build_visual_command(model: ModelRecord, profile: ProfileRecord, ctx: RuntimeContext) -> list[str]:
    paths = engine_paths(model, profile, ctx)
    command = [
        str(Path(ctx.edge_build) / "examples" / "multimodal" / "visual_build"),
        "--onnxDir",
        str(paths.model_root / "onnx" / "visual"),
        "--engineDir",
        str(paths.multimodal_dir),
    ]
    if profile.visual.get("maxImageTokens") is not None:
        command.extend(["--maxImageTokens", str(profile.visual["maxImageTokens"])])
    if profile.visual.get("maxImageTokensPerImage") is not None:
        command.extend(["--maxImageTokensPerImage", str(profile.visual["maxImageTokensPerImage"])])
    return command


def _run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    rendered = shlex.join(command)
    if dry_run:
        print(f"DRY-RUN  {rendered}")
        return
    subprocess.run(command, check=True, env=env)


def cmd_list(args: argparse.Namespace) -> int:
    models, profiles, _ = load_registries()
    ctx = _runtime_context()
    active: dict[str, Any] | None = None
    try:
        active = read_active_state(Path(ctx.state_file))
    except ModelCtlError:
        active = None
    for model in models.values():
        prep = prepared_status(model, ctx)
        print(f"{model.model_id} ({model.display_name}) prepared={prep['prepared']}")
        for profile in profiles.values():
            ok, _ = validate_engine_profile(model, profile, ctx, require_manifest=profile.managed)
            suffix = " active" if active and active.get("model_id") == model.model_id and active.get("engine_profile_id") == profile.profile_id else ""
            print(f"  - {profile.profile_id}: built={ok}{suffix}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    models, profiles, _ = load_registries()
    ctx = _runtime_context()
    model = resolve_model(args.model, models)
    prep = prepared_status(model, ctx)
    print(f"model_id: {model.model_id}")
    print(f"model_name: {model.display_name}")
    print(f"checkpoint_ready: {prep['checkpoint']}")
    print(f"quantized_ready: {prep['quantized']}")
    print(f"onnx_ready: {prep['onnx']}")
    if args.profile:
        profile = resolve_profile(args.profile, profiles)
        paths = engine_paths(model, profile, ctx)
        valid, errors = validate_engine_profile(model, profile, ctx, require_manifest=profile.managed)
        print(f"profile_id: {profile.profile_id}")
        print(f"engine_root: {paths.runtime_root}")
        print(f"managed_root: {paths.metadata_root}")
        print(f"engine_ready: {valid}")
        print(f"active: {is_active(model, profile, ctx)}")
        if errors:
            for error in errors:
                print(f"error: {error}")
    else:
        for profile in profiles.values():
            valid, _ = validate_engine_profile(model, profile, ctx, require_manifest=profile.managed)
            print(f"profile.{profile.profile_id}.engine_ready: {valid}")
    return 0


def _prepare_command(model: ModelRecord, *, dry_run: bool) -> list[str]:
    return [
        str(PREPARE_SCRIPT_PATH),
        *( ["--dry-run"] if dry_run else [] ),
        "--skip-edge-llm",
        "--skip-rtdetr",
        "--skip-data",
        "--skip-engine-build",
        "--skip-runtime-config",
        "--model-name",
        model.display_name,
    ]


def cmd_prepare(args: argparse.Namespace) -> int:
    models, _, _ = load_registries()
    model = resolve_model(args.model, models)
    ctx = _runtime_context()
    prep = prepared_status(model, ctx)
    if prep["prepared"]:
        print(f"Prepared artifacts already available for {model.model_id} at {model_root(model, ctx)}")
        return 0
    command = _prepare_command(model, dry_run=args.dry_run)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    print(f"Running: {shlex.join(command)}")
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    models, profiles, _ = load_registries()
    ctx = _runtime_context()
    model = resolve_model(args.model, models)
    profile = resolve_profile(args.profile, profiles)
    if not profile.managed:
        raise ModelCtlError(
            f"Profile {profile.profile_id} is a legacy adoption profile and is not rebuilt by modelctl."
        )
    prep = prepared_status(model, ctx)
    if not prep["prepared"]:
        raise ModelCtlError(
            f"Prepared artifacts are missing for {model.model_id}. Run `./scripts/modelctl prepare {model.model_id}` first."
        )
    if _managed_engine_valid(model, profile, ctx):
        print(f"Managed engine already built and validated for {model.model_id}/{profile.profile_id}")
        return 0
    paths = engine_paths(model, profile, ctx)
    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = ctx.plugin_path
    if not args.dry_run:
        Path(paths.llm_dir).mkdir(parents=True, exist_ok=True)
        Path(paths.multimodal_dir).mkdir(parents=True, exist_ok=True)
        llm_builder = Path(ctx.edge_build) / "examples" / "llm" / "llm_build"
        visual_builder = Path(ctx.edge_build) / "examples" / "multimodal" / "visual_build"
        if not llm_builder.exists():
            raise ModelCtlError(f"Missing llm_build executable at {llm_builder}")
        if not visual_builder.exists():
            raise ModelCtlError(f"Missing visual_build executable at {visual_builder}")
        if not Path(ctx.plugin_path).exists():
            raise ModelCtlError(f"Missing Edge-LLM plugin at {ctx.plugin_path}")
    _run_command(build_llm_command(model, profile, ctx), env=env, dry_run=args.dry_run)
    _run_command(build_visual_command(model, profile, ctx), env=env, dry_run=args.dry_run)
    if args.dry_run:
        print(f"DRY-RUN  would write engine manifest {paths.manifest_path}")
        return 0
    ok, errors = validate_engine_profile(model, profile, ctx, require_manifest=False)
    if not ok:
        raise ModelCtlError("\n".join(errors))
    _write_json_atomic(paths.manifest_path, _engine_manifest_payload(model, profile, ctx))
    ok, errors = validate_engine_profile(model, profile, ctx, require_manifest=True)
    if not ok:
        raise ModelCtlError("\n".join(errors))
    print(f"Built {model.model_id}/{profile.profile_id} at {paths.runtime_root}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    models, profiles, _ = load_registries()
    ctx = _runtime_context()
    model = resolve_model(args.model, models)
    profile = resolve_profile(args.profile, profiles)
    ok, errors = validate_engine_profile(model, profile, ctx, require_manifest=profile.managed)
    if not ok:
        raise ModelCtlError("\n".join(errors))
    print(f"Validated {model.model_id}/{profile.profile_id}")
    return 0


def is_active(model: ModelRecord, profile: ProfileRecord, ctx: RuntimeContext) -> bool:
    try:
        payload = read_active_state(Path(ctx.state_file))
    except ModelCtlError:
        return False
    return payload.get("model_id") == model.model_id and payload.get("engine_profile_id") == profile.profile_id


def cmd_activate(args: argparse.Namespace) -> int:
    models, profiles, _ = load_registries()
    ctx = _runtime_context()
    model = resolve_model(args.model, models)
    profile = resolve_profile(args.profile, profiles)
    ok, errors = validate_engine_profile(model, profile, ctx, require_manifest=profile.managed)
    if not ok:
        raise ModelCtlError("Cannot activate an unvalidated profile:\n" + "\n".join(errors))
    if args.dry_run:
        print(f"DRY-RUN  write active state {ctx.state_file}")
        print(f"DRY-RUN  write runtime env wrapper {ctx.env_file}")
        return 0
    ensure_runtime_env_file(ctx, dry_run=False)
    _write_json_atomic(Path(ctx.state_file), _active_state_payload(model, profile, ctx))
    print(f"Activated {model.model_id}/{profile.profile_id}")
    return 0


def cmd_current(args: argparse.Namespace) -> int:
    payload = read_active_state()
    print(f"model_id: {payload['model_id']}")
    print(f"model_name: {payload['model_name']}")
    print(f"profile_id: {payload['engine_profile_id']}")
    print(f"llm_engine_dir: {payload['llm_engine_dir']}")
    print(f"multimodal_engine_dir: {payload['multimodal_engine_dir']}")
    print(f"plugin_path: {payload['plugin_path']}")
    return 0


def cmd_print_env(args: argparse.Namespace) -> int:
    payload = read_active_state(Path(args.state_file))
    exports = [
        _shell_export("EDGE_VLM_MODEL_NAME", payload["model_name"]),
        _shell_export("EDGE_VLM_MODEL_ID", payload["model_id"]),
        _shell_export("EDGE_VLM_ENGINE_PROFILE_ID", payload["engine_profile_id"]),
        _shell_export("EDGE_VLM_LLM_ENGINE_DIR", payload["llm_engine_dir"]),
        _shell_export("EDGE_VLM_MULTIMODAL_ENGINE_DIR", payload["multimodal_engine_dir"]),
        _shell_export("EDGELLM_PLUGIN_PATH", payload["plugin_path"]),
    ]
    sys.stdout.write("\n".join(exports) + "\n")
    return 0


def cmd_registry_check(args: argparse.Namespace) -> int:
    models, profiles, manifest = load_registries()
    print(f"models={len(models)} profiles={len(profiles)} manifest_default={manifest['default_model']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registry-driven Thor model/profile management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    status = subparsers.add_parser("status")
    status.add_argument("model")
    status.add_argument("profile", nargs="?")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("model")
    prepare.add_argument("--dry-run", action="store_true")

    build = subparsers.add_parser("build")
    build.add_argument("model")
    build.add_argument("profile")
    build.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("model")
    validate.add_argument("profile")

    activate = subparsers.add_parser("activate")
    activate.add_argument("model")
    activate.add_argument("profile")
    activate.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("current")

    print_env = subparsers.add_parser("print-env")
    print_env.add_argument("--state-file", required=True)

    subparsers.add_parser("registry-check")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        match args.command:
            case "list":
                return cmd_list(args)
            case "status":
                return cmd_status(args)
            case "prepare":
                return cmd_prepare(args)
            case "build":
                return cmd_build(args)
            case "validate":
                return cmd_validate(args)
            case "activate":
                return cmd_activate(args)
            case "current":
                return cmd_current(args)
            case "print-env":
                return cmd_print_env(args)
            case "registry-check":
                return cmd_registry_check(args)
            case _:
                parser.error(f"unknown command {args.command}")
    except ModelCtlError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
