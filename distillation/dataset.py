"""
Deterministic dataset build, split, and export for the distillation pipeline.

The canonical dataset is kept framework-neutral (list of TemporalSamples).
``export_sft_dataset`` converts accepted samples into the SFT/QLoRA chat
format expected by the student training launcher.

Split
-----
``split_dataset`` produces deterministic train/validation/test splits from
a random seed.  It groups samples by their ``sample_id`` prefix (everything
before the last ``-`` separator) to avoid related control variants leaking
across splits.

Manifest
--------
Every exported dataset directory gets a ``dataset_manifest.json`` that
records sample IDs, content hashes, source provenance, and exporter version.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from distillation.schema import TemporalSample

EXPORTER_VERSION: str = "sft_exporter_v1"

SFT_SYSTEM_PROMPT: str = (
    "You are a compact temporal observation model. "
    "Given ordered frames from a sensor sequence, return a structured JSON "
    "observation about any detected temporal change."
)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _group_key(sample_id: str) -> str:
    """Extract a grouping key to prevent related variants from leaking across splits."""
    # Use everything before the last '-segment' as the group key.
    parts = sample_id.rsplit("-", 1)
    return parts[0] if len(parts) > 1 else sample_id


def split_dataset(
    samples: List[TemporalSample],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[TemporalSample], List[TemporalSample], List[TemporalSample]]:
    """
    Split accepted samples into (train, validation, test) deterministically.

    Related samples (same ``_group_key``) are kept in the same split to avoid
    leakage.

    Parameters
    ----------
    samples    : list of accepted TemporalSamples
    train_frac : fraction for training (default 0.70)
    val_frac   : fraction for validation (default 0.15)
                 test_frac is implicitly ``1 - train_frac - val_frac``
    seed       : random seed for reproducibility

    Returns
    -------
    (train, val, test) as separate lists.
    """
    assert 0 < train_frac < 1
    assert 0 < val_frac < 1
    assert train_frac + val_frac < 1.0

    # Group by key.
    groups: Dict[str, List[TemporalSample]] = {}
    for s in samples:
        k = _group_key(s.sample_id)
        groups.setdefault(k, []).append(s)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train: n_train + n_val])
    # remainder -> test

    train: List[TemporalSample] = []
    val: List[TemporalSample] = []
    test: List[TemporalSample] = []

    for key in keys:
        group = groups[key]
        if key in train_keys:
            train.extend(group)
        elif key in val_keys:
            val.extend(group)
        else:
            test.extend(group)

    return train, val, test


# ---------------------------------------------------------------------------
# SFT export
# ---------------------------------------------------------------------------

_SUPPORTED_VIDEO_ENCODINGS = frozenset({
    "video_tensor",
    # #74 native Qwen3-VL/Cosmos-Reason2 video path with MRoPE timestamps
    "native_qwen3vl_video_imagedata_mrope_timestamps",
})
_SUPPORTED_VIDEO_SEQUENCE_TYPES = frozenset({
    "video",
    # #74 temporal_images: ordered frames delivered to Qwen3-VL as a native
    # video ImageData object (not independent per-frame images).
    "temporal_images",
})
_SUPPORTED_IMAGE_SEQUENCE_TYPES = frozenset({"image_sequence", ""})


def _sample_to_sft_example(sample: TemporalSample) -> dict:
    """Convert one TemporalSample to a multimodal SFT chat example.

    The export format uses the processor-native Qwen3-VL / Cosmos-Reason2
    content schema (``type: "image"`` / ``type: "video"``) rather than the
    OpenAI ``image_url`` wrapper so that ``processor.apply_chat_template`` can
    resolve media directly without a custom URL resolver.

    * Native-video samples (``sequence_type`` in ``_SUPPORTED_VIDEO_SEQUENCE_TYPES``
      or ``runtime_temporal_encoding`` in ``_SUPPORTED_VIDEO_ENCODINGS``):
      exported as ``{"type": "video", "path": [frame0, frame1, ...]}``.
      Transformers' Qwen3-VL processor accepts a list of sampled frame paths
      under this key and builds the ``[T,H,W,3]`` video tensor automatically.

    * Image-sequence / legacy samples: exported as N independent
      ``{"type": "image", "url": <path>}`` entries in chronological order,
      compatible with the Qwen3-VL independent-images processor path.

    Raises
    ------
    ValueError
        If a video-sequence sample has no frames.
    AssertionError
        If the sample has no target.
    """
    assert sample.target is not None, f"Sample {sample.sample_id} has no target"

    prov = sample.provenance
    is_video = (
        prov.sequence_type in _SUPPORTED_VIDEO_SEQUENCE_TYPES
        or prov.runtime_temporal_encoding in _SUPPORTED_VIDEO_ENCODINGS
    )

    user_content_parts: list = []

    if is_video:
        # Native-video path: export as {"type": "video", "path": [list of frame
        # paths]} which is the Qwen3-VL / Transformers-native representation.
        # The processor resolves the list of paths into a [T,H,W,3] video tensor
        # and handles MRoPE temporal position IDs automatically.
        if len(sample.frames) == 0:
            raise ValueError(
                f"Sample {sample.sample_id!r} has a video sequence_type but no frames."
            )
        frame_paths = [fr.path for fr in sample.frames]
        video_content: dict = {"type": "video", "path": frame_paths}
        if prov.effective_fps > 0:
            video_content["fps"] = prov.effective_fps
        # Carry per-frame timestamps as metadata so training logs / evaluators
        # can reconstruct the temporal layout without re-reading the files.
        video_content["t_seconds"] = [fr.t_seconds for fr in sample.frames]
        user_content_parts.append(video_content)
    else:
        # Independent-images path (image_sequence or legacy/unspecified).
        # Export as {"type": "image", "url": <path>} — the Qwen3-VL native
        # per-image content schema consumed by AutoProcessor/apply_chat_template.
        for fr in sample.frames:
            user_content_parts.append({"type": "image", "url": fr.path})

    user_content_parts.append(
        {
            "type": "text",
            "text": (
                f"Analyze this {len(sample.frames)}-frame temporal sequence "
                f"(frames in chronological order). "
                "Return a JSON object with the temporal observation."
            ),
        }
    )
    assistant_content = json.dumps(sample.target.to_dict(), sort_keys=True)
    return {
        "sample_id": sample.sample_id,
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content_parts},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "prompt_profile": sample.prompt_profile,
            "frame_count": sample.frame_count(),
            "provenance": sample.provenance.to_dict(),
            "content_hash": sample.content_hash(),
            # Record the effective export modality so consumers can verify
            # train/inference representation consistency.
            "export_modality": "video" if is_video else "image_sequence",
        },
    }


def export_sft_dataset(
    train: List[TemporalSample],
    val: List[TemporalSample],
    test: List[TemporalSample],
    output_dir: Path,
    source_manifest_hash: str = "",
    repo_commit: str = "",
) -> dict:
    """
    Export train/val/test splits to ``output_dir`` in SFT chat format.

    Writes three JSONL files and a ``dataset_manifest.json``.

    Returns the manifest dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest: dict = {
        "exporter_version": EXPORTER_VERSION,
        "repo_commit": repo_commit,
        "source_manifest_hash": source_manifest_hash,
        "splits": {},
    }

    for split_name, split_samples in [("train", train), ("val", val), ("test", test)]:
        split_path = output_dir / f"{split_name}.jsonl"
        lines = []
        sample_records = []
        for s in split_samples:
            example = _sample_to_sft_example(s)
            lines.append(json.dumps(example))
            sample_records.append(
                {"sample_id": s.sample_id, "content_hash": s.content_hash()}
            )
        split_path.write_text("\n".join(lines))
        split_hash = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
        split_manifest["splits"][split_name] = {
            "count": len(split_samples),
            "file": split_path.name,
            "hash": split_hash,
            "samples": sample_records,
        }

    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2))
    return split_manifest
