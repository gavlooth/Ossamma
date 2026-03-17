#!/usr/bin/env python3
"""
Generate raw teacher responses for REBEL distillation requests.

This script consumes the JSONL emitted by `build_rebel_teacher_requests.jl` and
produces a resumable JSONL of raw teacher responses. The output is intended to
feed `parse_rebel_teacher_responses.jl`.

Example:
    python3 scripts/generate_rebel_teacher_responses.py \
        --input data/rebel/train_teacher_requests.jsonl \
        --output data/rebel/train_teacher_raw.jsonl \
        --teacher-model Qwen/Qwen2.5-7B-Instruct \
        --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw teacher responses for REBEL requests.")
    parser.add_argument("--input", required=True, help="Input request JSONL")
    parser.add_argument("--output", required=True, help="Output raw response JSONL")
    parser.add_argument(
        "--teacher-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Teacher model id/path",
    )
    parser.add_argument(
        "--teacher-revision",
        default="",
        help="Optional teacher revision/hash for reproducibility",
    )
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="Field containing the rendered prompt text",
    )
    parser.add_argument(
        "--system-prompt-field",
        default="system_prompt",
        help="Field containing the system prompt",
    )
    parser.add_argument(
        "--response-field",
        default="response",
        help="Field name to store the raw teacher response",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip match keys already present in the output JSONL",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output JSONL when not using --resume",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of deterministic shards",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This process shard id in [0, num_shards)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap after sharding and resume filtering",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=2048,
        help="Prompt truncation length before generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum teacher completion length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for sampling",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--stop-sequence",
        action="append",
        default=[],
        help="Optional stop sequence; may be repeated",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Teacher device: auto | cpu | cuda | cuda:0",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Teacher dtype",
    )
    parser.add_argument(
        "--plain-prompt",
        action="store_true",
        help="Disable chat template usage even if the tokenizer supports it",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF model/tokenizer loading",
    )
    parser.add_argument(
        "--allow-cpu-teacher",
        action="store_true",
        help="Allow teacher generation on CPU-only PyTorch runtimes; intended for tiny-model smoke tests only",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of prompts to generate in parallel (default 16)",
    )
    return parser.parse_args()


def import_runtime():
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required for teacher generation but was not found. "
            "Install `torch` in the active Python environment before running this script."
        ) from exc

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "transformers is required for teacher generation but could not be imported."
        ) from exc

    return torch, AutoModelForCausalLM, AutoTokenizer


@dataclass
class RequestRow:
    source_index: int
    prompt: str
    system_prompt: str
    row: Dict[str, object]


def iter_request_rows(path: Path, prompt_field: str, system_prompt_field: str) -> Iterator[RequestRow]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get(prompt_field, "")).strip()
            if not prompt:
                continue
            system_prompt = str(row.get(system_prompt_field, "")).strip()
            yield RequestRow(
                source_index=idx,
                prompt=prompt,
                system_prompt=system_prompt,
                row=row,
            )


def has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def resolve_dtype(torch, dtype_name: str):
    if dtype_name == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def build_rendered_prompt(tokenizer, prompt: str, system_prompt: str, plain_prompt: bool) -> str:
    if not plain_prompt and hasattr(tokenizer, "apply_chat_template"):
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    return prompt


def apply_stop_sequences(text: str, stop_sequences: List[str]) -> str:
    if not stop_sequences:
        return text
    cutoff = len(text)
    for stop in stop_sequences:
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0:
            cutoff = min(cutoff, index)
    return text[:cutoff].rstrip()


def shard_filter(source_index: int, num_shards: int, shard_id: int) -> bool:
    return (source_index % num_shards) == shard_id


def load_existing_match_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            match_key = row.get("match_key")
            if isinstance(match_key, str) and match_key:
                keys.add(match_key)
    return keys


def hparam_dict(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision or None,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": bool(args.do_sample),
        "stop_sequences": list(args.stop_sequence),
        "device": args.device,
        "dtype": args.dtype,
        "plain_prompt": bool(args.plain_prompt),
        "allow_cpu_teacher": bool(args.allow_cpu_teacher),
    }


def looks_like_smoke_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return (
        "tiny" in lowered
        or "micro" in lowered
        or "small" in lowered
        or lowered.endswith("gpt2")
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must satisfy 0 <= shard-id < num-shards")
    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --resume or --overwrite.")

    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime()
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available and not args.allow_cpu_teacher and not looks_like_smoke_model(args.teacher_model):
        raise SystemExit(
            "CPU-only PyTorch runtime detected. Refusing to run a non-smoke teacher model on CPU. "
            "Install CUDA-enabled PyTorch or rerun with --allow-cpu-teacher for explicit smoke testing."
        )

    if args.overwrite and output_path.exists() and not args.resume:
        output_path.unlink()
    if args.overwrite and error_path.exists() and not args.resume:
        error_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_match_keys(output_path) if args.resume else set()

    tokenizer_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.teacher_revision:
        tokenizer_kwargs["revision"] = args.teacher_revision
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, **tokenizer_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": resolve_dtype(torch, args.dtype),
    }
    if args.teacher_revision:
        model_kwargs["revision"] = args.teacher_revision
    if args.device == "auto" and has_accelerate():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **model_kwargs)
    if args.device == "auto" and "device_map" not in model_kwargs:
        runtime_device = "cuda" if cuda_available else "cpu"
        model.to(runtime_device)
    elif args.device != "auto":
        runtime_device = args.device
        model.to(runtime_device)
    else:
        # device_map="auto" places model on GPU; resolve actual device for input tensors
        runtime_device = "cuda" if cuda_available else "cpu"
    model.eval()

    accepted = 0
    skipped_existing = 0
    skipped_shard = 0
    failed = 0
    generation_cfg = hparam_dict(args)
    mode = "a" if args.resume else "w"
    batch_size = max(1, args.batch_size)

    # Collect pending requests
    pending: List[tuple] = []  # (row_dict, match_key, rendered_prompt)
    for request in iter_request_rows(input_path, args.prompt_field, args.system_prompt_field):
        if not shard_filter(request.source_index, args.num_shards, args.shard_id):
            skipped_shard += 1
            continue
        row = dict(request.row)
        match_key = str(row.get("match_key", "")).strip()
        if not match_key:
            match_key = f"row_index:{request.source_index}"
            row["match_key"] = match_key
        if match_key in existing_keys:
            skipped_existing += 1
            continue
        rendered_prompt = build_rendered_prompt(
            tokenizer, request.prompt, request.system_prompt, args.plain_prompt,
        )
        pending.append((row, match_key, rendered_prompt))
        if args.max_rows > 0 and len(pending) + len(existing_keys) >= args.max_rows:
            break

    import time
    total_pending = len(pending)
    t0 = time.time()
    print(f"Generating {total_pending} responses in batches of {batch_size}...", flush=True)

    with output_path.open(mode, encoding="utf-8") as out_handle, error_path.open(mode, encoding="utf-8") as err_handle:
        for batch_start in range(0, total_pending, batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            batch_prompts = [item[2] for item in batch]

            try:
                tokenizer.padding_side = "left"
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_tokens,
                    padding=True,
                )
                input_ids = encoded["input_ids"].to(runtime_device)
                attention_mask = encoded["attention_mask"].to(runtime_device)
                prompt_lengths = attention_mask.sum(dim=1).tolist()

                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                for i, (row, match_key, _prompt) in enumerate(batch):
                    try:
                        prompt_len = int(prompt_lengths[i])
                        new_tokens = generated[i][input_ids.shape[1]:]
                        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                        response_text = apply_stop_sequences(response_text, args.stop_sequence)
                        if not response_text:
                            raise ValueError("teacher generation returned empty text")

                        record = dict(row)
                        record[args.response_field] = response_text
                        record["teacher_model"] = args.teacher_model
                        record["teacher_revision"] = args.teacher_revision or None
                        record["generation_config"] = generation_cfg
                        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        existing_keys.add(match_key)
                        accepted += 1
                    except Exception as exc:
                        failed += 1
                        err_handle.write(
                            json.dumps(
                                {"match_key": match_key, "source_index": batch_start + i, "error": repr(exc)},
                                ensure_ascii=False,
                            ) + "\n"
                        )
                out_handle.flush()
                err_handle.flush()
            except Exception as exc:
                # Whole batch failed — log each item
                for i, (row, match_key, _prompt) in enumerate(batch):
                    failed += 1
                    err_handle.write(
                        json.dumps(
                            {"match_key": match_key, "source_index": batch_start + i, "error": repr(exc)},
                            ensure_ascii=False,
                        ) + "\n"
                    )
                err_handle.flush()

            done = min(batch_start + batch_size, total_pending)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_pending - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total_pending}] {rate:.1f} rows/s, ETA {eta/60:.0f}m", flush=True)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "error_log": str(error_path),
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision or None,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "accepted": accepted,
        "skipped_existing": skipped_existing,
        "skipped_shard": skipped_shard,
        "failed": failed,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
