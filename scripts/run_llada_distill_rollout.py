#!/usr/bin/env python3
"""
Run the locked offline PRIME/LLaDA distillation rollout.

This wrapper keeps the first 3B rollout reproducible:
1. Load repo-tracked prompt and sampling manifests
2. Prepare the offline teacher-distilled corpus
3. Run the canonical trainer in --prepare-only mode as a hard gate
4. Only launch training when explicitly requested
5. Copy the locked manifests and generated corpus manifest into run directories
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PRIME/LLaDA offline distillation rollout.")
    parser.add_argument(
        "--config",
        default="configs/llada_prime_3b_distill_offline.toml",
        help="Student config TOML",
    )
    parser.add_argument(
        "--prompt-manifest",
        default="configs/distillation/llada_prime_3b_prompt_manifest.json",
        help="Repo-tracked prompt/build manifest JSON",
    )
    parser.add_argument(
        "--sampling-manifest",
        default="configs/distillation/llada_prime_3b_qwen25_7b_sampling_manifest.json",
        help="Repo-tracked teacher sampling manifest JSON",
    )
    parser.add_argument("--raw-input", default="", help="Override raw source corpus path")
    parser.add_argument(
        "--output-dir",
        default="data/distill/llada_prime_3b",
        help="Prepared distillation corpus directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/llada_prime_3b_distill_pilot",
        help="Checkpoint/output directory for the student run",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--julia", default="julia", help="Julia executable to use")
    parser.add_argument("--teacher-model", default="", help="Override teacher model id/path")
    parser.add_argument("--teacher-revision", default="", help="Override teacher revision/hash")
    parser.add_argument("--tokenizer-model", default="", help="Optional student tokenizer override")
    parser.add_argument("--seq-len", type=int, default=-1, help="Optional trainer seq_len override")
    parser.add_argument("--stride", type=int, default=-1, help="Optional trainer stride override")
    parser.add_argument("--batch-size", type=int, default=-1, help="Optional trainer batch_size override")
    parser.add_argument("--total-steps", type=int, default=-1, help="Optional trainer total_steps override")
    parser.add_argument("--eval-every", type=int, default=-1, help="Optional trainer eval_every override")
    parser.add_argument("--save-every", type=int, default=-1, help="Optional trainer save_every override")
    parser.add_argument("--log-every", type=int, default=-1, help="Optional trainer log_every override")
    parser.add_argument("--learning-rate", type=float, default=-1.0, help="Optional trainer learning_rate override")
    parser.add_argument("--sample-steps", type=int, default=-1, help="Optional trainer sample_steps override")
    parser.add_argument("--max-train-texts", type=int, default=-1, help="Optional cap for trainer prep/train")
    parser.add_argument("--max-val-texts", type=int, default=-1, help="Optional cap for trainer prep/train")
    parser.add_argument("--max-prompts", type=int, default=-1, help="Optional prompt count override")
    parser.add_argument("--prefix-words", type=int, default=-1, help="Optional prompt prefix length override")
    parser.add_argument("--min-words", type=int, default=-1, help="Optional minimum source words override")
    parser.add_argument("--val-ratio", type=float, default=-1.0, help="Optional validation ratio override")
    parser.add_argument("--max-new-tokens", type=int, default=-1, help="Optional teacher completion override")
    parser.add_argument("--temperature", type=float, default=-1.0, help="Optional teacher temperature override")
    parser.add_argument("--top-p", type=float, default=-1.0, help="Optional teacher top-p override")
    parser.add_argument("--device", default="", help="Optional teacher device override")
    parser.add_argument("--dtype", default="", help="Optional teacher dtype override")
    parser.add_argument("--plain-prompt", action="store_true", help="Force plain prompt rendering")
    parser.add_argument("--resume-generation", action="store_true", help="Resume teacher generation if teacher.jsonl already exists")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated corpus artifacts")
    parser.add_argument("--resume-checkpoint", default="", help="Optional student checkpoint to resume from")
    parser.add_argument("--launch-training", action="store_true", help="Launch student training after the prep gate passes")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def extend_arg(cmd: List[str], flag: str, value: Any, *, positive_only: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, (int, float)):
        if positive_only and value <= 0:
            return
    if isinstance(value, str) and not value:
        return
    cmd.extend([flag, str(value)])


def copy_manifests(paths: Iterable[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, destination_dir / path.name)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    python_runtime = shutil.which(args.python) or str(Path(args.python).resolve())

    config_path = resolve_path(root, args.config)
    prompt_manifest_path = resolve_path(root, args.prompt_manifest)
    sampling_manifest_path = resolve_path(root, args.sampling_manifest)
    output_dir = resolve_path(root, args.output_dir)
    checkpoint_dir = resolve_path(root, args.checkpoint_dir)

    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    if not prompt_manifest_path.exists():
        raise SystemExit(f"Prompt manifest not found: {prompt_manifest_path}")
    if not sampling_manifest_path.exists():
        raise SystemExit(f"Sampling manifest not found: {sampling_manifest_path}")

    prompt_manifest = load_json(prompt_manifest_path)
    sampling_manifest = load_json(sampling_manifest_path)

    raw_input_value = args.raw_input or str(prompt_manifest.get("raw_input", "")).strip()
    if not raw_input_value:
        raise SystemExit("No raw input provided. Set --raw-input or raw_input in the prompt manifest.")
    raw_input_path = resolve_path(root, raw_input_value)
    if not raw_input_path.exists():
        raise SystemExit(f"Raw input not found: {raw_input_path}")

    builder_cfg = dict(prompt_manifest.get("builder", {}))
    packaging_cfg = dict(prompt_manifest.get("packaging", {}))
    runtime_cfg = dict(sampling_manifest.get("teacher_runtime", {}))
    generation_cfg = dict(sampling_manifest.get("generation", {}))
    shard_cfg = dict(sampling_manifest.get("sharding", {}))

    teacher_model = args.teacher_model or str(sampling_manifest.get("teacher_model", "")).strip()
    teacher_revision = args.teacher_revision or str(sampling_manifest.get("teacher_revision", "")).strip()
    if not teacher_model:
        raise SystemExit("No teacher model locked. Set it in the sampling manifest or via --teacher-model.")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    locked_manifest_dir = output_dir / "locked_manifests"
    copy_manifests((prompt_manifest_path, sampling_manifest_path, config_path), locked_manifest_dir)

    prepare_cmd = [
        args.python,
        str(root / "scripts" / "prepare_llada_distill_corpus.py"),
        "--raw-input",
        str(raw_input_path),
        "--output-dir",
        str(output_dir),
        "--text-field",
        str(prompt_manifest.get("text_field", "text")),
        "--alt-text-field",
        str(prompt_manifest.get("alt_text_field", "content")),
        "--teacher-model",
        teacher_model,
        "--render-mode",
        str(packaging_cfg.get("render_mode", "prompt_response")),
        "--task",
        str(prompt_manifest.get("task", "continuation")),
        "--system-prompt",
        str(prompt_manifest.get("system_prompt", "")),
        "--seed",
        str(builder_cfg.get("seed", 42)),
        "--max-prompts",
        str(args.max_prompts if args.max_prompts > 0 else builder_cfg.get("max_prompts", 5000)),
        "--prefix-words",
        str(args.prefix_words if args.prefix_words > 0 else builder_cfg.get("prefix_words", 48)),
        "--min-words",
        str(args.min_words if args.min_words > 0 else builder_cfg.get("min_words", 80)),
        "--val-ratio",
        str(args.val_ratio if args.val_ratio > 0 else builder_cfg.get("val_ratio", 0.05)),
        "--max-input-tokens",
        str(generation_cfg.get("max_input_tokens", 2048)),
        "--max-new-tokens",
        str(args.max_new_tokens if args.max_new_tokens > 0 else generation_cfg.get("max_new_tokens", 512)),
        "--temperature",
        str(args.temperature if args.temperature > 0 else generation_cfg.get("temperature", 0.7)),
        "--top-p",
        str(args.top_p if args.top_p > 0 else generation_cfg.get("top_p", 0.9)),
        "--device",
        args.device or str(runtime_cfg.get("device", "auto")),
        "--dtype",
        args.dtype or str(runtime_cfg.get("dtype", "auto")),
        "--num-shards",
        str(shard_cfg.get("num_shards", 1)),
        "--shard-id",
        str(shard_cfg.get("shard_id", 0)),
        "--teacher-repeat",
        str(packaging_cfg.get("teacher_repeat", 1)),
        "--raw-repeat",
        str(packaging_cfg.get("raw_repeat", 1)),
    ]
    if teacher_revision:
        prepare_cmd.extend(["--teacher-revision", teacher_revision])
    if generation_cfg.get("do_sample", False):
        prepare_cmd.append("--do-sample")
    for stop_sequence in generation_cfg.get("stop_sequences", []):
        prepare_cmd.extend(["--stop-sequence", str(stop_sequence)])
    if runtime_cfg.get("trust_remote_code", False):
        prepare_cmd.append("--trust-remote-code")
    if args.plain_prompt or runtime_cfg.get("plain_prompt", False):
        prepare_cmd.append("--plain-prompt")
    if args.resume_generation:
        prepare_cmd.append("--resume")
    if args.overwrite:
        prepare_cmd.append("--overwrite")

    raw_train = str(packaging_cfg.get("raw_train", "")).strip()
    raw_val = str(packaging_cfg.get("raw_val", "")).strip()
    if raw_train:
        prepare_cmd.extend(["--raw-train", str(resolve_path(root, raw_train))])
    if raw_val:
        prepare_cmd.extend(["--raw-val", str(resolve_path(root, raw_val))])

    run_command(prepare_cmd, root)

    corpus_manifest_path = output_dir / "manifest.json"
    if not corpus_manifest_path.exists():
        raise SystemExit(f"Expected corpus manifest was not produced: {corpus_manifest_path}")

    provenance_dir = checkpoint_dir / "run_manifests"
    copy_manifests((prompt_manifest_path, sampling_manifest_path, config_path, corpus_manifest_path), provenance_dir)

    teacher_mix = packaging_cfg.get("teacher_mixture_ratio")
    raw_mix = packaging_cfg.get("raw_mixture_ratio")
    if teacher_mix is None or raw_mix is None:
        raw_enabled = bool(raw_train or raw_val)
        teacher_mix = 1.0 if not raw_enabled else 0.5
        raw_mix = 0.0 if not raw_enabled else 0.5

    trainer_base = [
        args.julia,
        "--project=.",
        "scripts/train_llada_canonical.jl",
        "--config",
        str(config_path),
        "--train-path",
        str(output_dir / "train.jsonl"),
        "--val-path",
        str(output_dir / "validation.jsonl"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--distillation-mode",
        "offline_sequence",
        "--teacher-model",
        teacher_model,
        "--corpus-manifest",
        str(corpus_manifest_path),
        "--teacher-mixture-ratio",
        str(teacher_mix),
        "--raw-mixture-ratio",
        str(raw_mix),
    ]
    if teacher_revision:
        trainer_base.extend(["--teacher-revision", teacher_revision])
    extend_arg(trainer_base, "--tokenizer-model", args.tokenizer_model)
    extend_arg(trainer_base, "--seq-len", args.seq_len, positive_only=True)
    extend_arg(trainer_base, "--stride", args.stride, positive_only=True)
    extend_arg(trainer_base, "--batch-size", args.batch_size, positive_only=True)
    extend_arg(trainer_base, "--total-steps", args.total_steps, positive_only=True)
    extend_arg(trainer_base, "--eval-every", args.eval_every, positive_only=True)
    extend_arg(trainer_base, "--save-every", args.save_every, positive_only=True)
    extend_arg(trainer_base, "--log-every", args.log_every, positive_only=True)
    extend_arg(trainer_base, "--learning-rate", args.learning_rate, positive_only=True)
    extend_arg(trainer_base, "--sample-steps", args.sample_steps, positive_only=True)
    extend_arg(trainer_base, "--max-train-texts", args.max_train_texts, positive_only=True)
    extend_arg(trainer_base, "--max-val-texts", args.max_val_texts, positive_only=True)
    extend_arg(trainer_base, "--resume", args.resume_checkpoint)

    prep_cmd = trainer_base + ["--prepare-only"]
    julia_env = dict(os.environ)
    julia_env["PYCALL_JL_RUNTIME_PYTHON"] = python_runtime
    julia_env.setdefault("CRYPTOGRAPHY_OPENSSL_NO_LEGACY", "1")
    run_command(prep_cmd, root, env=julia_env)

    rollout_summary = {
        "config": str(config_path),
        "prompt_manifest": str(prompt_manifest_path),
        "sampling_manifest": str(sampling_manifest_path),
        "raw_input": str(raw_input_path),
        "prepared_output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "python_runtime": python_runtime,
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision or None,
        "launch_training": bool(args.launch_training),
        "corpus_manifest": str(corpus_manifest_path),
    }
    with (checkpoint_dir / "rollout_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rollout_summary, handle, indent=2, ensure_ascii=False)

    if args.launch_training:
        run_command(trainer_base, root, env=julia_env)

    print(json.dumps(rollout_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
