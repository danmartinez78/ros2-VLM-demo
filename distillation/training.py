"""
Student training config and dry-run launcher for Cosmos-Reason2-2B SFT/QLoRA.

The launcher deliberately does *not* import torch, transformers, or peft so
that dry-run / config-validation passes in CPU-only CI without any model
downloads.

Usage
-----
    python -m distillation.training --config training/configs/cr2_2b_lora_v1.yaml --dry-run

or from Python:

    from distillation.training import TrainingConfig, run_training
    cfg = TrainingConfig.from_yaml(path)
    run_training(cfg, dry_run=True)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

TRAINING_CONFIG_VERSION: str = "training_config_v1"


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def to_dict(self) -> dict:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoraConfig":
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass
class TrainingConfig:
    """
    Versioned configuration for a CR2-2B SFT/QLoRA experiment.

    All paths are expressed relative to a ``base_dir`` (default: repo root)
    so that no machine-specific absolute paths are committed.
    """

    config_version: str = TRAINING_CONFIG_VERSION
    experiment_id: str = "cr2-2b-lora-exp001"

    # Model
    base_model: str = "nvidia/Cosmos-Reason2-2B"
    base_checkpoint: str = ""
    """Optional local checkpoint path (relative).  Empty = download from hub."""

    # Dataset
    dataset_dir: str = "distillation/datasets/sft_v1"
    dataset_manifest_hash: str = ""

    # Output
    output_dir: str = "distillation/runs/{experiment_id}"
    resume_from_checkpoint: str = ""
    """Set to a checkpoint dir to resume; empty = start fresh."""

    # Training hyperparameters
    num_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 10
    max_seq_length: int = 2048
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 0
    seed: int = 42

    # LoRA
    use_lora: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Provenance (filled at runtime)
    repo_commit: str = ""
    base_checkpoint_hash: str = ""

    # NOTE: no GRPO/RL fields – reserved for a follow-on implementation.

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "config_version": self.config_version,
            "experiment_id": self.experiment_id,
            "base_model": self.base_model,
            "base_checkpoint": self.base_checkpoint,
            "dataset_dir": self.dataset_dir,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "output_dir": self.output_dir,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "num_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "dataloader_num_workers": self.dataloader_num_workers,
            "seed": self.seed,
            "use_lora": self.use_lora,
            "lora": self.lora.to_dict(),
            "repo_commit": self.repo_commit,
            "base_checkpoint_hash": self.base_checkpoint_hash,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        d = dict(d)  # shallow copy – do not mutate caller's dict
        lora_d = d.pop("lora", {})
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.lora = LoraConfig.from_dict(lora_d)
        return cfg

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load from a YAML file (requires PyYAML; falls back to JSON)."""
        text = path.read_text()
        try:
            import yaml  # type: ignore[import-untyped]
            d = yaml.safe_load(text)
        except ImportError:
            d = json.loads(text)
        return cls.from_dict(d)

    def to_yaml(self, path: Path) -> None:
        d = self.to_dict()
        try:
            import yaml  # type: ignore[import-untyped]
            path.write_text(yaml.dump(d, default_flow_style=False))
        except ImportError:
            path.write_text(json.dumps(d, indent=2))

    def resolved_output_dir(self) -> str:
        return self.output_dir.format(experiment_id=self.experiment_id)

    def config_hash(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Return a list of validation error strings.  Empty list means valid.
        """
        errors: List[str] = []
        if not self.experiment_id.strip():
            errors.append("experiment_id must not be empty")
        if not self.base_model.strip():
            errors.append("base_model must not be empty")
        if not self.dataset_dir.strip():
            errors.append("dataset_dir must not be empty")
        if not self.output_dir.strip():
            errors.append("output_dir must not be empty")
        if self.num_epochs < 1:
            errors.append("num_epochs must be >= 1")
        if self.per_device_train_batch_size < 1:
            errors.append("per_device_train_batch_size must be >= 1")
        if not (0.0 < self.learning_rate < 1.0):
            errors.append(f"learning_rate {self.learning_rate} looks unreasonable")
        if not (0.0 <= self.lora.dropout <= 1.0):
            errors.append(f"lora.dropout {self.lora.dropout} out of range [0,1]")
        if self.fp16 and self.bf16:
            errors.append("fp16 and bf16 cannot both be True")
        # Warn if absolute path crept in
        for attr in ("base_checkpoint", "dataset_dir", "output_dir"):
            val = getattr(self, attr)
            if val and os.path.isabs(val):
                errors.append(
                    f"{attr} is an absolute path ({val!r}); use a relative path instead"
                )
        return errors


# ---------------------------------------------------------------------------
# Dry-run launcher
# ---------------------------------------------------------------------------

def run_training(
    config: TrainingConfig,
    dry_run: bool = False,
    output_dir_override: Optional[Path] = None,
) -> dict:
    """
    Validate the config and either launch or dry-run the training job.

    Parameters
    ----------
    config            : validated TrainingConfig
    dry_run           : if True, only validate and emit the plan; no model loads
    output_dir_override : override output dir (for tests)

    Returns
    -------
    A dict with keys:
      - "dry_run": bool
      - "valid": bool
      - "errors": list of error strings
      - "plan": dict describing the resolved config and output paths
    """
    errors = config.validate()
    result: dict = {
        "dry_run": dry_run,
        "valid": len(errors) == 0,
        "errors": errors,
        "plan": {},
    }

    if errors:
        return result

    resolved_out = output_dir_override or Path(config.resolved_output_dir())
    plan = {
        "experiment_id": config.experiment_id,
        "config_version": config.config_version,
        "base_model": config.base_model,
        "dataset_dir": config.dataset_dir,
        "output_dir": str(resolved_out),
        "num_epochs": config.num_epochs,
        "use_lora": config.use_lora,
        "lora": config.lora.to_dict(),
        "seed": config.seed,
        "config_hash": config.config_hash(),
        "resume_from_checkpoint": config.resume_from_checkpoint or None,
    }
    result["plan"] = plan

    if dry_run:
        # Write the effective config for provenance without loading the model.
        resolved_out.mkdir(parents=True, exist_ok=True)
        effective_cfg_path = resolved_out / "effective_config.json"
        effective_cfg_path.write_text(json.dumps(config.to_dict(), indent=2))
        return result

    # Real training path (not executed in CI – requires GPU + model download).
    raise NotImplementedError(
        "Real training requires GPU hardware and model downloads. "
        "Use --dry-run for CI validation."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="CR2-2B SFT/QLoRA training launcher"
    )
    parser.add_argument("--config", required=True, type=Path, help="Training config YAML/JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and emit a plan without loading the model",
    )
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    result = run_training(cfg, dry_run=args.dry_run)

    print(json.dumps(result, indent=2))
    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    _main()
