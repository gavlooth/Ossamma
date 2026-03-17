#!/usr/bin/env python3
"""
Generate a teacher corpus for PRIME/LLaDA distillation.

This script reads prompts from JSONL or plain text and writes one JSONL row per
teacher completion. It is designed for resumable, sharded offline sequence
distillation, which matches the current canonical PRIME training path.

Example:
    python3 scripts/generate_llada_teacher_corpus.py \
        --input prompts/train.jsonl \
        --output data/distill/teacher_train.jsonl \
        --teacher-model Qwen/Qwen2.5-7B-Instruct \
        --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher corpus for PRIME/LLaDA distillation.")
    parser.add_argument("--input", required=True, help="Prompt source (.jsonl or .txt)")
    parser.add_argument("--output", required=True, help="Output teacher JSONL path")
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
        help="JSONL field to read as the prompt",
    )
    parser.add_argument(
        "--split-field",
        default="split",
        help="JSONL field containing the source split",
    )
    parser.add_argument(
        "--task-field",
        default="task",
        help="JSONL field containing the source task/category",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="JSONL field containing an optional prompt id",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt prepended or injected through chat templates",
    )
    parser.add_argument(
        "--plain-prompt",
        action="store_true",
        help="Disable chat template usage even if the tokenizer supports it",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompt hashes already present in the output JSONL",
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
        "--start-index",
        type=int,
        default=0,
        help="Inclusive starting source index after sharding",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="Exclusive ending source index after sharding; <=0 means no cap",
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
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p for sampling",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy/beam-like decoding",
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
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF model/tokenizer loading",
    )
    return parser.parse_args()


def import_runtime():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency check
        raise SystemExit(
            "PyTorch is required for teacher generation but was not found. "
            "Install `torch` in the active Python environment before running this script."
        ) from exc

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency check
        raise SystemExit(
            "transformers is required for teacher generation but could not be imported."
        ) from exc

    return torch, AutoModelForCausalLM, AutoTokenizer


@dataclass
class PromptRow:
    source_index: int
    prompt: str
    source_id: Optional[str]
    source_split: str
    source_task: str
    system_prompt: str
    raw: Dict[str, object]


def iter_prompt_rows(
    path: Path,
    prompt_field: str,
    split_field: str,
    task_field: str,
    id_field: str,
) -> Iterator[PromptRow]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = str(row.get(prompt_field, "")).strip()
                if not prompt:
                    continue
                source_id = row.get(id_field)
                yield PromptRow(
                    source_index=idx,
                    prompt=prompt,
                    source_id=None if source_id is None else str(source_id),
                    source_split=str(row.get(split_field, "train")),
                    source_task=str(row.get(task_field, "general")),
                    system_prompt=str(row.get("system_prompt", "")),
                    raw=row,
                )
        return

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            prompt = line.strip()
            if not prompt:
                continue
            yield PromptRow(
                source_index=idx,
                prompt=prompt,
                source_id=str(idx),
                source_split="train",
                source_task="general",
                system_prompt="",
                raw={},
            )


def compute_prompt_hash(prompt: str, system_prompt: str) -> str:
    payload = json.dumps(
        {"prompt": prompt, "system_prompt": system_prompt},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_existing_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_hash = row.get("prompt_hash")
            if isinstance(prompt_hash, str) and prompt_hash:
                hashes.add(prompt_hash)
    return hashes


def resolve_dtype(torch, dtype_name: str):
    if dtype_name == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


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


def shard_filter(source_index: int, num_shards: int, shard_id: int) -> bool:
    return (source_index % num_shards) == shard_id


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
        "system_prompt": args.system_prompt,
        "plain_prompt": bool(args.plain_prompt),
        "device": args.device,
        "dtype": args.dtype,
    }


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
        raise SystemExit(
            f"Output already exists: {output_path}. Use --resume or --overwrite."
        )

    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime()

    if args.overwrite and output_path.exists() and not args.resume:
        output_path.unlink()
    if args.overwrite and error_path.exists() and not args.resume:
        error_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_hashes = load_existing_hashes(output_path) if args.resume else set()

    tokenizer_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.teacher_revision:
        tokenizer_kwargs["revision"] = args.teacher_revision
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, **tokenizer_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": resolve_dtype(torch, args.dtype),
    }
    if args.teacher_revision:
        model_kwargs["revision"] = args.teacher_revision
    if args.device == "auto" and has_accelerate():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **model_kwargs)
    if args.device == "auto" and "device_map" not in model_kwargs:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(target_device)
        runtime_device = target_device
    elif args.device != "auto":
        model.to(args.device)
        runtime_device = args.device
    else:
        runtime_device = args.device
    model.eval()

    accepted = 0
    skipped_existing = 0
    skipped_shard = 0
    failed = 0

    generation_cfg = hparam_dict(args)
    mode = "a" if args.resume else "w"

    with output_path.open(mode, encoding="utf-8") as out_handle, error_path.open(mode, encoding="utf-8") as err_handle:
        for row in iter_prompt_rows(
            input_path,
            prompt_field=args.prompt_field,
            split_field=args.split_field,
            task_field=args.task_field,
            id_field=args.id_field,
        ):
            if not shard_filter(row.source_index, args.num_shards, args.shard_id):
                skipped_shard += 1
                continue

            local_index = row.source_index // args.num_shards
            if local_index < args.start_index:
                continue
            if args.end_index > 0 and local_index >= args.end_index:
                break

            effective_system_prompt = row.system_prompt or args.system_prompt
            prompt_hash = compute_prompt_hash(row.prompt, effective_system_prompt)
            if prompt_hash in existing_hashes:
                skipped_existing += 1
                continue

            rendered_prompt = build_rendered_prompt(
                tokenizer,
                row.prompt,
                effective_system_prompt,
                args.plain_prompt,
            )

            try:
                encoded = tokenizer(
                    rendered_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_tokens,
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")

                if runtime_device != "auto":
                    input_ids = input_ids.to(runtime_device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(runtime_device)

                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                prompt_len = int(input_ids.shape[-1])
                new_tokens = generated[0][prompt_len:]
                teacher_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                teacher_text = apply_stop_sequences(teacher_text, args.stop_sequence)
                if not teacher_text:
                    raise ValueError("teacher generation returned empty text")

                record = {
                    "prompt_hash": prompt_hash,
                    "prompt": row.prompt,
                    "teacher_text": teacher_text,
                    "teacher_model": args.teacher_model,
                    "teacher_revision": args.teacher_revision or None,
                    "generation_config": generation_cfg,
                    "system_prompt": effective_system_prompt,
                    "source_index": row.source_index,
                    "source_id": row.source_id,
                    "source_split": row.source_split,
                    "source_task": row.source_task,
                }
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_handle.flush()
                existing_hashes.add(prompt_hash)
                accepted += 1
            except Exception as exc:  # pragma: no cover - runtime failure capture
                failed += 1
                err_handle.write(
                    json.dumps(
                        {
                            "prompt_hash": prompt_hash,
                            "source_index": row.source_index,
                            "source_id": row.source_id,
                            "prompt": row.prompt,
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                err_handle.flush()

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
