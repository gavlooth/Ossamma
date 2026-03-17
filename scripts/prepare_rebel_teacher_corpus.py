#!/usr/bin/env python3
"""
End-to-end teacher corpus preparation for REBEL distillation.

Pipeline:
1. Build teacher request JSONL from base REBEL rows
2. Generate raw teacher responses
3. Parse raw responses into normalized teacher annotation rows
4. Merge teacher annotations back into the base REBEL rows
5. Validate the merged corpus
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare REBEL teacher corpus end-to-end.")
    parser.add_argument("--input", required=True, help="Base REBEL JSONL input")
    parser.add_argument("--output-dir", required=True, help="Output directory for pipeline artifacts")
    parser.add_argument("--raw-input", default="", help="Existing raw teacher response JSONL to reuse")
    parser.add_argument(
        "--teacher-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Teacher model id/path",
    )
    parser.add_argument("--teacher-revision", default="", help="Optional teacher revision/hash")
    parser.add_argument("--device", default="auto", help="Teacher runtime device")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Teacher dtype",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for smoke runs")
    parser.add_argument("--max-input-tokens", type=int, default=2048, help="Teacher prompt truncation length")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum teacher completion length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top-p")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--plain-prompt", action="store_true", help="Disable chat template rendering")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for HF loading")
    parser.add_argument(
        "--allow-cpu-teacher",
        action="store_true",
        help="Allow teacher generation on CPU-only runtimes; intended for tiny-model smoke tests only",
    )
    parser.add_argument("--resume", action="store_true", help="Resume teacher generation if raw output already exists")
    parser.add_argument("--prompt-style", default="compact", choices=("verbose", "compact"), help="Prompt style for teacher requests")
    parser.add_argument(
        "--relation-label-mode",
        default="id_or_name",
        choices=("id", "id_or_name", "name"),
        help="Relation label format in prompts",
    )
    parser.add_argument("--strict", action="store_true", help="Enable strict parse/merge/validate checks")
    parser.add_argument("--skip-generation", action="store_true", help="Skip teacher generation and reuse existing raw responses")
    parser.add_argument(
        "--allow-empty-teacher",
        action="store_true",
        help="Do not require parsed teacher coverage during final validation; useful for runtime smoke tests",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requests_path = output_dir / "teacher_requests.jsonl"
    raw_path = Path(args.raw_input).resolve() if args.raw_input else (output_dir / "teacher_raw.jsonl")
    parsed_path = output_dir / "teacher_parsed.jsonl"
    merged_path = output_dir / "teacher_merged.jsonl"

    build_cmd = [
        "julia", "--project=.",
        "scripts/build_rebel_teacher_requests.jl",
        "--input", str(input_path),
        "--output", str(requests_path),
        "--prompt-style", args.prompt_style,
        "--relation-label-mode", args.relation_label_mode,
    ]
    if args.max_rows > 0:
        build_cmd.extend(["--max-rows", str(args.max_rows)])
    run(build_cmd, root)

    if not args.skip_generation:
        gen_cmd = [
            sys.executable,
            "scripts/generate_rebel_teacher_responses.py",
            "--input", str(requests_path),
            "--output", str(raw_path),
            "--teacher-model", args.teacher_model,
            "--device", args.device,
            "--dtype", args.dtype,
            "--max-input-tokens", str(args.max_input_tokens),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
        ]
        if args.teacher_revision:
            gen_cmd.extend(["--teacher-revision", args.teacher_revision])
        if args.max_rows > 0:
            gen_cmd.extend(["--max-rows", str(args.max_rows)])
        if args.do_sample:
            gen_cmd.append("--do-sample")
        if args.plain_prompt:
            gen_cmd.append("--plain-prompt")
        if args.trust_remote_code:
            gen_cmd.append("--trust-remote-code")
        if args.allow_cpu_teacher:
            gen_cmd.append("--allow-cpu-teacher")
        if args.resume:
            gen_cmd.append("--resume")
        run(gen_cmd, root)
    elif not raw_path.exists():
        raise SystemExit(f"--skip-generation was set but raw input was not found: {raw_path}")

    parse_cmd = [
        "julia", "--project=.",
        "scripts/parse_rebel_teacher_responses.jl",
        "--input", str(raw_path),
        "--output", str(parsed_path),
    ]
    if args.strict:
        parse_cmd.append("--strict")
    if args.max_rows > 0:
        parse_cmd.extend(["--max-rows", str(args.max_rows)])
    run(parse_cmd, root)

    merge_cmd = [
        "julia", "--project=.",
        "scripts/merge_rebel_teacher_targets.jl",
        "--base", str(input_path),
        "--teacher", str(parsed_path),
        "--output", str(merged_path),
    ]
    if args.strict:
        merge_cmd.append("--strict")
    run(merge_cmd, root)

    validate_cmd = [
        "julia", "--project=.",
        "scripts/validate_rebel_teacher_targets.jl",
        "--data", str(merged_path),
    ]
    if not args.allow_empty_teacher:
        validate_cmd.append("--require-teacher")
    if args.strict:
        validate_cmd.append("--strict")
    run(validate_cmd, root)

    print("============================================================")
    print("REBEL Teacher Corpus Prepared")
    print("============================================================")
    print(f"Requests: {requests_path}")
    print(f"Raw:      {raw_path}")
    print(f"Parsed:   {parsed_path}")
    print(f"Merged:   {merged_path}")
    print(f"Require teacher coverage: {not args.allow_empty_teacher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
