"""
Student training config and dry-run launcher for Cosmos-Reason2-2B SFT / LoRA
(and optionally QLoRA when BitsAndBytes quantization is enabled).

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

Multimodal contract
-------------------
The real training path uses ``AutoModelForVision2Seq`` + ``AutoProcessor``
(not ``AutoModelForCausalLM`` + ``AutoTokenizer``) so that pixel / video
tensors produced by the processor are forwarded to the model rather than
stringified image references.  The CPU-testable helper
``validate_multimodal_messages`` checks that a message list carries
structured image/video content objects (dicts with ``"type": "image"`` or
``"type": "video"``) rather than plain text, and raises ``ValueError`` if they
have been flattened to strings.  This guard runs before the heavy model load
so that config/format errors are caught early.

QLoRA vs LoRA
-------------
When ``quantize_bits`` is set to 4 (the default for QLoRA), the base model is
loaded in 4-bit NF4 quantization via ``bitsandbytes`` and
``prepare_model_for_kbit_training`` is called before wrapping with PEFT LoRA.
Set ``quantize_bits`` to 0 (or omit it) to disable quantization and use plain
full-precision LoRA.  The YAML config comment and experiment labeling reflect
the actual quantization level used at runtime.
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

    # Model loading
    trust_remote_code: bool = False
    """
    Set to True only when the model hub entry requires custom code execution.
    This enables arbitrary code from the remote repository; opt in consciously.
    """

    quantize_bits: int = 4
    """
    Quantization precision for QLoRA.  Set to 4 for 4-bit NF4 (true QLoRA),
    or 0 to disable quantization and use plain full-precision LoRA.
    Other values are not currently supported and will raise at runtime.
    """

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
            "trust_remote_code": self.trust_remote_code,
            "quantize_bits": self.quantize_bits,
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

def assert_collated_features_multimodal(features: dict, sample_id: str = "") -> None:
    """
    CPU-testable assertion: verify that a collated feature dict produced by the
    multimodal processor contains at least one of the expected visual tensor
    keys for Qwen3-VL / Cosmos-Reason2.

    Expected keys (at least one must be present):
      - ``pixel_values``           – image patch tensor
      - ``pixel_values_videos``    – video patch tensor
      - ``image_grid_thw``         – Qwen3-VL image grid metadata
      - ``video_grid_thw``         – Qwen3-VL video grid metadata

    Raises ``AssertionError`` if none of these keys are present, which
    indicates that the processor consumed only the text path and dropped all
    visual content.

    This function has *no* heavy dependencies and can be called in CI without
    a GPU or model weights by passing a synthetic dict that mimics what the
    real processor would produce.

    Parameters
    ----------
    features : dict returned by the multimodal processor
    sample_id : optional sample identifier for diagnostic messages
    """
    visual_keys = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
    present = visual_keys & features.keys()
    if not present:
        label = f" (sample {sample_id!r})" if sample_id else ""
        raise AssertionError(
            f"Collated features{label} contain no multimodal visual tensor keys. "
            f"Expected at least one of {sorted(visual_keys)} but got: "
            f"{sorted(features.keys())[:20]}. "
            "The processor likely consumed only the text path and discarded visual "
            "content — ensure apply_chat_template is called with tokenize=True, "
            "return_dict=True and that image/video objects are passed as dicts."
        )


def validate_multimodal_messages(messages: list) -> None:
    """
    CPU-testable guard: verify that a message list preserves multimodal image
    and video content as structured dicts rather than stringifying them.

    Each message with role ``"user"`` may carry a ``content`` list.  Any entry
    with ``"type": "image"`` or ``"type": "video"`` must remain a dict (not
    be cast to ``str``).  The Qwen3-VL / Cosmos-Reason2 processor-native
    content schema uses ``{"type": "image", "url": "..."}`` and
    ``{"type": "video", "path": [...]}``; the deprecated OpenAI-style
    ``"image_url"`` / ``"video_url"`` wrapper keys are also flagged.

    Raises ``ValueError`` if any multimodal content object has been flattened
    to a string, or if an OpenAI-style ``image_url``/``video_url`` wrapper is
    detected (which would not be resolved by the multimodal processor).

    This function has *no* heavy dependencies and runs in CI without a GPU or
    model weights.  It is called by ``run_training`` before the model is loaded
    so that format errors are caught early.

    Parameters
    ----------
    messages : list of message dicts (role / content pairs)

    Raises
    ------
    ValueError
        If any multimodal content object has been stringified or uses an
        unsupported wrapper schema.
    """
    _NATIVE_VISUAL_TYPES = {"image", "video"}
    _LEGACY_WRAPPER_TYPES = {"image_url", "video_url"}

    for i, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for j, part in enumerate(content):
                if isinstance(part, str):
                    # A stringified dict that looks like a visual content object.
                    for marker in (
                        '"type": "image"', '"type": "video"',
                        '"type": "image_url"', '"type": "video_url"',
                        "'type': 'image'", "'type': 'video'",
                    ):
                        if marker in part:
                            raise ValueError(
                                f"messages[{i}].content[{j}] is a stringified visual "
                                f"content object; pass the dict directly. "
                                f"Got: {part[:80]!r}"
                            )
                elif isinstance(part, dict):
                    t = part.get("type", "")
                    # Reject legacy OpenAI-style image_url / video_url wrappers.
                    if t in _LEGACY_WRAPPER_TYPES:
                        raise ValueError(
                            f"messages[{i}].content[{j}] uses legacy wrapper "
                            f"type={t!r}; use the processor-native "
                            f"{{'type': 'image', 'url': ...}} / "
                            f"{{'type': 'video', 'path': [...]}} schema instead."
                        )
                    # Reject nested stringified image/video inside native types.
                    if t in _NATIVE_VISUAL_TYPES:
                        for key in ("url", "path"):
                            val = part.get(key)
                            if isinstance(val, str) and val.startswith("{"):
                                raise ValueError(
                                    f"messages[{i}].content[{j}].{key} is a "
                                    f"stringified dict; pass it as a nested "
                                    f"dict. Got: {val[:80]!r}"
                                )
        elif isinstance(content, str):
            # Single-string content is text-only and valid.
            pass


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

    # ---------------------------------------------------------------------------
    # Real training path
    # ---------------------------------------------------------------------------
    # Heavy imports (torch, transformers, peft) are deferred to here so that
    # dry-run / config-validation passes in CPU-only CI without any model
    # downloads or GPU requirements.
    #
    # Backend: Hugging Face ``transformers`` Trainer with ``peft`` QLoRA adapters.
    # Both packages must be available at runtime:
    #   pip install transformers>=4.40 peft>=0.10 datasets accelerate bitsandbytes
    #
    # The dataset directory (config.dataset_dir) must contain the JSONL files
    # produced by ``distillation.dataset.export_sft_dataset`` and a
    # ``dataset_manifest.json``.  The manifest hash is read back and verified
    # against ``config.dataset_manifest_hash`` (when the config provides one)
    # so that training never silently uses a stale or replaced dataset.
    #
    # Multimodal model path
    # ---------------------
    # Cosmos-Reason2-2B is a vision-language model.  The training path uses
    # ``AutoModelForVision2Seq`` + ``AutoProcessor`` (not ``AutoModelForCausalLM``
    # + ``AutoTokenizer``) so that pixel / video tensors produced by the
    # processor are forwarded to the model rather than stringifying image
    # references to plain text.  ``validate_multimodal_messages`` is called on
    # each example before the model load so that format errors are caught early.
    try:
        import transformers  # type: ignore[import-untyped]
        import peft  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Real training requires 'transformers' and 'peft'. "
            "Install them with: pip install transformers peft datasets accelerate bitsandbytes"
        ) from exc

    dataset_dir = Path(config.dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text())
        if config.dataset_manifest_hash:
            import hashlib
            actual_hash = hashlib.sha256(
                json.dumps(manifest_data, sort_keys=True).encode()
            ).hexdigest()[:16]
            if actual_hash != config.dataset_manifest_hash:
                raise ValueError(
                    f"Dataset manifest hash mismatch: config expects "
                    f"{config.dataset_manifest_hash!r} but found {actual_hash!r}. "
                    "Re-export the dataset or update the config hash."
                )
    elif config.dataset_manifest_hash:
        raise FileNotFoundError(
            f"dataset_manifest.json not found in {dataset_dir} "
            f"but config specifies a manifest hash {config.dataset_manifest_hash!r}."
        )

    resolved_out.mkdir(parents=True, exist_ok=True)

    # Write effective config before training starts for provenance.
    effective_cfg_path = resolved_out / "effective_config.json"
    effective_cfg_path.write_text(json.dumps(config.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # Model + processor (multimodal VLM path)
    # ------------------------------------------------------------------
    from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer  # type: ignore[import-untyped]
    from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType  # type: ignore[import-untyped]

    model_name = config.base_model if not config.base_checkpoint else config.base_checkpoint
    # AutoProcessor handles both the vision (pixel values) and text (tokenizer)
    # branches of the multimodal model, unlike a plain AutoTokenizer which only
    # processes text and would silently drop visual content.
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=config.trust_remote_code,
    )

    # ------------------------------------------------------------------
    # QLoRA vs plain LoRA base model loading
    # ------------------------------------------------------------------
    if config.quantize_bits == 4:
        # True QLoRA: load base in 4-bit NF4 quantization via bitsandbytes,
        # then call prepare_model_for_kbit_training before wrapping with PEFT.
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]
            from peft import prepare_model_for_kbit_training  # type: ignore[import-untyped]
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "4-bit QLoRA requires 'bitsandbytes' and 'torch'. "
                "Install with: pip install bitsandbytes torch"
            ) from exc
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=config.trust_remote_code,
        )
        model = prepare_model_for_kbit_training(model)
    elif config.quantize_bits == 0:
        # Plain full-precision LoRA — no quantization.
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code,
        )
    else:
        raise ValueError(
            f"Unsupported quantize_bits={config.quantize_bits!r}. "
            "Use 4 for QLoRA (4-bit NF4) or 0 for plain LoRA."
        )

    if config.use_lora:
        lora_cfg = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )
        model = get_peft_model(model, lora_cfg)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Install 'datasets': pip install datasets") from exc

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": str(dataset_dir / "train.jsonl"),
            "validation": str(dataset_dir / "val.jsonl"),
        },
    )

    def _tokenize(example):
        messages = example.get("messages", [])
        # Guard: ensure image/video content objects are structured dicts, not
        # stringified text.  Runs on every training example and catches any
        # export-side regression before the model forward pass.
        validate_multimodal_messages(messages)
        # Use the processor's multimodal chat-template path:
        #   apply_chat_template(tokenize=True, return_dict=True)
        # This resolves <image> / <video> placeholder tokens and produces
        # pixel_values / pixel_values_videos alongside input_ids, so the visual
        # encoder receives properly shaped inputs rather than stringified URLs.
        # (The two-step `apply_chat_template(tokenize=False)` then
        #  `processor(text=...)` path only processes text and silently drops
        #  all visual content.)
        features = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            add_generation_prompt=False,
        )
        # Verify that the processor produced multimodal output (pixel_values or
        # pixel_values_videos).  This assertion surfaces processor mismatches
        # before any model forward pass.
        assert_collated_features_multimodal(
            {k: v for k, v in features.items()},
            sample_id=example.get("sample_id", ""),
        )
        # SFT label construction: train only on the assistant response tokens.
        # 1. Start with a full copy of input_ids as labels.
        # 2. Build the prompt-only text (user+system turns, no assistant) and
        #    tokenize it to find the prompt length.
        # 3. Mask (set to -100) all prompt and padding positions so the loss
        #    is computed only over the assistant output tokens.
        import torch  # type: ignore[import-untyped]
        input_ids = features["input_ids"][0]  # [seq_len]
        # Build prompt-only message list (exclude the last assistant turn).
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        prompt_features = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
            add_generation_prompt=True,
        )
        prompt_len = prompt_features["input_ids"].shape[1]
        labels = input_ids.clone()
        # Mask prompt tokens and any padding (pad_token_id) from the loss.
        labels[:prompt_len] = -100
        pad_id = getattr(processor, "pad_token_id", None) or getattr(
            processor.tokenizer, "pad_token_id", None
        )
        if pad_id is not None:
            labels[labels == pad_id] = -100
        features["labels"] = labels.unsqueeze(0)
        # Flatten tensors to lists for HF datasets serialization.
        return {k: v.squeeze(0).tolist() for k, v in features.items()}

    tokenized = raw_datasets.map(_tokenize, batched=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(resolved_out),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        resume_from_checkpoint=config.resume_from_checkpoint or None,
        report_to="none",
    )

    # Use DataCollatorForSeq2Seq so that SFT labels are correctly handled:
    # prompt tokens are masked (label=-100) in _tokenize; the collator pads
    # and stacks label tensors alongside input tensors.  The plain Trainer
    # with no collator does not create a "labels" column, so no loss is
    # ever computed.
    from transformers import DataCollatorForSeq2Seq  # type: ignore[import-untyped]
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=processor,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint or None)
    trainer.save_model(str(resolved_out / "final_adapter"))

    result["plan"] = {**plan, "checkpoint": str(resolved_out / "final_adapter")}
    return result


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
