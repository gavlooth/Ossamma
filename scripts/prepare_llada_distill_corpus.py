#!/usr/bin/env python3
"""
End-to-end offline corpus preparation for PRIME/LLaDA distillation.

Pipeline:
1. Build prompts from raw corpus
2. Generate teacher completions
3. Validate/clean teacher rows
4. Package cleaned rows for scripts/train_llada_canonical.jl
5. Emit a manifest JSON for the run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare offline PRIME/LLaDA distillation corpus.")
    parser.add_argument("--raw-input", required=True, help="Raw source corpus (.txt or .jsonl)")
    parser.add_argument("--output-dir", required=True, help="Output directory for prepared corpus")
    parser.add_argument("--text-field", default="text", help="Primary text field when reading JSONL raw corpus")
    parser.add_argument("--alt-text-field", default="content", help="Fallback text field when reading JSONL raw corpus")
    parser.add_argument(
        "--teacher-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Teacher model id/path",
    )
    parser.add_argument(
        "--teacher-revision",
        default="",
        help="Optional teacher revision/hash",
    )
    parser.add_argument(
        "--render-mode",
        choices=("prompt_response", "response_only"),
        default="prompt_response",
        help="Packaging mode for final student corpus",
    )
    parser.add_argument(
        "--task",
        choices=("continuation",),
        default="continuation",
        help="Prompt family to build",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a precise, fluent assistant. Continue the user's text naturally and coherently.",
        help="System prompt stored in prompt manifests and used by generation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-prompts", type=int, default=5000, help="Maximum prompts to build")
    parser.add_argument("--prefix-words", type=int, default=48, help="Words used in each continuation prompt")
    parser.add_argument("--min-words", type=int, default=80, help="Minimum source text length in words")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio for prompt-building and fallback validation split")
    parser.add_argument("--max-input-tokens", type=int, default=2048, help="Teacher prompt truncation length")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Teacher completion length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Teacher generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Teacher top-p")
    parser.add_argument("--do-sample", action="store_true", help="Enable teacher sampling")
    parser.add_argument("--stop-sequence", action="append", default=[], help="Optional teacher stop sequence; may be repeated")
    parser.add_argument("--resume", action="store_true", help="Resume generation if teacher.jsonl already exists")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--plain-prompt", action="store_true", help="Disable chat-template rendering")
    parser.add_argument("--device", default="auto", help="Teacher runtime device passed through to generation")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto", help="Teacher runtime dtype")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow HF trust_remote_code during teacher loading")
    parser.add_argument("--num-shards", type=int, default=1, help="Total generation shards")
    parser.add_argument("--shard-id", type=int, default=0, help="This process shard id")
    parser.add_argument("--teacher-repeat", type=int, default=1, help="Teacher row repeat factor in packaged corpus")
    parser.add_argument("--raw-train", default="", help="Optional raw train corpus to mix into final training set")
    parser.add_argument("--raw-val", default="", help="Optional raw val corpus to mix into final validation set")
    parser.add_argument("--raw-repeat", type=int, default=1, help="Raw row repeat factor in packaged corpus")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = output_dir / "prompts.jsonl"
    teacher_path = output_dir / "teacher.jsonl"
    clean_train_path = output_dir / "clean_train.jsonl"
    clean_val_path = output_dir / "clean_val.jsonl"
    stats_path = output_dir / "teacher_stats.json"
    packaged_train_path = output_dir / "train.jsonl"
    packaged_val_path = output_dir / "validation.jsonl"
    manifest_path = output_dir / "manifest.json"

    if args.overwrite:
        for path in (
            prompts_path,
            teacher_path,
            teacher_path.with_suffix(".jsonl.errors.jsonl"),
            clean_train_path,
            clean_val_path,
            stats_path,
            packaged_train_path,
            packaged_val_path,
            manifest_path,
        ):
            if path.exists():
                path.unlink()

    python = sys.executable
    root = Path(__file__).resolve().parent

    run_command(
        [
            python,
            str(root / "build_llada_distill_prompts.py"),
            "--input",
            args.raw_input,
            "--output",
            str(prompts_path),
            "--text-field",
            args.text_field,
            "--alt-text-field",
            args.alt_text_field,
            "--task",
            args.task,
            "--system-prompt",
            args.system_prompt,
            "--seed",
            str(args.seed),
            "--max-prompts",
            str(args.max_prompts),
            "--prefix-words",
            str(args.prefix_words),
            "--min-words",
            str(args.min_words),
            "--val-ratio",
            str(args.val_ratio),
        ]
    )

    gen_cmd = [
        python,
        str(root / "generate_llada_teacher_corpus.py"),
        "--input",
        str(prompts_path),
        "--output",
        str(teacher_path),
        "--teacher-model",
        args.teacher_model,
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--num-shards",
        str(args.num_shards),
        "--shard-id",
        str(args.shard_id),
    ]
    if args.teacher_revision:
        gen_cmd.extend(["--teacher-revision", args.teacher_revision])
    if args.do_sample:
        gen_cmd.append("--do-sample")
    for stop_sequence in args.stop_sequence:
        gen_cmd.extend(["--stop-sequence", stop_sequence])
    if args.resume:
        gen_cmd.append("--resume")
    if args.overwrite:
        gen_cmd.append("--overwrite")
    if args.plain_prompt:
        gen_cmd.append("--plain-prompt")
    if args.trust_remote_code:
        gen_cmd.append("--trust-remote-code")
    run_command(gen_cmd)

    run_command(
        [
            python,
            str(root / "validate_llada_teacher_corpus.py"),
            "--input",
            str(teacher_path),
            "--output-train",
            str(clean_train_path),
            "--output-val",
            str(clean_val_path),
            "--stats-output",
            str(stats_path),
            "--val-ratio",
            str(args.val_ratio),
        ]
    )

    pkg_cmd = [
        python,
        str(root / "package_llada_distill_corpus.py"),
        "--teacher-train",
        str(clean_train_path),
        "--teacher-val",
        str(clean_val_path),
        "--output-train",
        str(packaged_train_path),
        "--output-val",
        str(packaged_val_path),
        "--manifest-output",
        str(manifest_path),
        "--render-mode",
        args.render_mode,
        "--teacher-repeat",
        str(args.teacher_repeat),
        "--raw-repeat",
        str(args.raw_repeat),
    ]
    if args.raw_train:
        pkg_cmd.extend(["--raw-train", args.raw_train])
    if args.raw_val:
        pkg_cmd.extend(["--raw-val", args.raw_val])
    run_command(pkg_cmd)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.update(
        {
            "teacher_model": args.teacher_model,
            "teacher_revision": args.teacher_revision or None,
            "raw_input": args.raw_input,
            "prompts_path": str(prompts_path),
            "teacher_path": str(teacher_path),
            "clean_train_path": str(clean_train_path),
            "clean_val_path": str(clean_val_path),
            "stats_path": str(stats_path),
            "text_field": args.text_field,
            "alt_text_field": args.alt_text_field,
            "task": args.task,
            "val_ratio": args.val_ratio,
            "system_prompt": args.system_prompt,
            "render_mode": args.render_mode,
            "stop_sequences": list(args.stop_sequence),
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
        }
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
