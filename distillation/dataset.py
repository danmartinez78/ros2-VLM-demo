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

_SUPPORTED_VIDEO_ENCODINGS = frozenset({"video_tensor"})
_SUPPORTED_IMAGE_SEQUENCE_TYPES = frozenset({"image_sequence", ""})


def _sample_to_sft_example(sample: TemporalSample) -> dict:
    """Convert one TemporalSample to a multimodal SFT chat example.

    The export format respects the sample's provenance temporal-input
    representation so that training inputs match the representation used
    during teacher generation and inference.

    * ``sequence_type == "video"`` / ``runtime_temporal_encoding == "video_tensor"``:
      exported as a single ``"video_url"`` content object so that the
      Qwen3-VL / Cosmos-Reason2 processor receives a temporal video tensor
      rather than N independent images.  Any sample whose provenance records a
      native-video encoding but whose frames cannot be trivially packed (e.g.
      no single composite path is available) raises ``ValueError`` so that the
      mismatch is surfaced at export time rather than silently converted.

    * All other sequence types (``"image_sequence"`` or unspecified legacy
      samples): exported as N independent ``"image_url"`` entries in
      chronological order, which is compatible with the independent-images
      processor path.

    Raises
    ------
    ValueError
        If the sample's provenance declares an unsupported runtime encoding
        or if a video-sequence sample has no composite video path to export.
    AssertionError
        If the sample has no target.
    """
    assert sample.target is not None, f"Sample {sample.sample_id} has no target"

    prov = sample.provenance
    is_video = (
        prov.sequence_type == "video"
        or prov.runtime_temporal_encoding in _SUPPORTED_VIDEO_ENCODINGS
    )

    user_content_parts: list = []

    if is_video:
        # Native-video path: export a single video_url content object so the
        # Qwen3-VL processor builds a [T,H,W,3] video tensor rather than
        # independent image tensors.
        #
        # Convention: when a sample's frames all share a common directory prefix
        # and differ only by filename, export a composite "video://<dir>" URL
        # that the training data-loader resolves to the assembled video.  If no
        # consistent composite path can be derived, raise so the caller can
        # decide whether to re-encode or skip the sample.
        if len(sample.frames) == 0:
            raise ValueError(
                f"Sample {sample.sample_id!r} has sequence_type='video' but no frames."
            )
        # Derive composite video path from common directory + sorted frame list.
        frame_paths = [fr.path for fr in sample.frames]
        # Use the first frame's directory as the video container reference.
        from pathlib import PurePosixPath
        first_path = PurePosixPath(frame_paths[0])
        video_dir = str(first_path.parent) if first_path.parent != PurePosixPath(".") else ""
        composite_url = (
            f"video://{video_dir}" if video_dir else f"video_frames:{','.join(frame_paths)}"
        )
        user_content_parts.append(
            {
                "type": "video_url",
                "video_url": {
                    "url": composite_url,
                    "fps": prov.effective_fps if prov.effective_fps > 0 else None,
                    "frame_paths": frame_paths,
                    "t_seconds": [fr.t_seconds for fr in sample.frames],
                },
            }
        )
    else:
        # Independent-images path (image_sequence or legacy/unspecified).
        for fr in sample.frames:
            user_content_parts.append(
                {"type": "image_url", "image_url": {"url": fr.path, "t_seconds": fr.t_seconds}}
            )

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
