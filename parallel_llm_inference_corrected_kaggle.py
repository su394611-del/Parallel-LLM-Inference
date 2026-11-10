"""Parallel LLM sentiment inference: one model replica and worker per GPU.

The input must be JSONL (one JSON object per line) with a nonempty text field,
which defaults to 'review'. Optional 'label' allows sentiment evaluation.

Examples (run as a .py SCRIPT, not directly in an ipykernel cell):

  # Fast offline multiprocessing check; NO model download or GPU required
  python parallel_llm_inference_corrected.py --mock --devices cpu cpu \
      --input reviews.jsonl --output results_mock.jsonl

  # On a single NVIDIA GPU (use a small batch on 4 GB VRAM)
  python parallel_llm_inference_corrected.py --devices cuda:0 \
      --input reviews.jsonl --output results_gpu.jsonl --batch-size 1

  # On a computer with two visible NVIDIA GPUs
  python parallel_llm_inference_corrected.py --devices cuda:0 cuda:1 \
      --input reviews.jsonl --output results_2gpu.jsonl --batch-size 2

This is independent-batch INFERENCE. It does not train or split one model
across multiple GPUs. Each GPU must have enough memory for a full model copy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import re
import signal
import time
import traceback
import threading
from typing import Any

import torch
from tqdm.auto import tqdm


POSITIVE_PHRASES = [
    "excellent", "fantastic", "amazing", "great", "wonderful", "reliable",
    "impressive", "love", "perfect", "smooth", "recommended", "satisfied",
]
NEGATIVE_PHRASES = [
    "terrible", "awful", "broken", "disappointed", "poor", "faulty",
    "unreliable", "hate", "worst", "stopped working", "failed", "frustrating",
]


def build_messages(text: str) -> list[dict[str, str]]:
    return [{
        "role": "user",
        "content": (
            "Classify this product review as positive or negative. "
            "Reply with exactly one word: positive or negative.\n\n"
            f"Review: {text}"
        ),
    }]


def make_demo_rows(n: int, seed: int = 2026) -> list[dict[str, Any]]:
    """Optional generated sample data when --input is omitted."""
    rng = random.Random(seed)
    components = ["battery", "display", "keyboard", "microphone", "touchpad", "fan", "speaker"]
    pos = ["is excellent", "works perfectly", "is reliable", "is impressive", "feels great"]
    neg = ["is terrible", "stopped working", "is unreliable", "is broken", "is disappointing"]
    rows = []
    for i in range(n):
        label = "positive" if i % 2 == 0 else "negative"
        phrases = pos if label == "positive" else neg
        count = rng.randint(1, 4)
        text = " ".join(f"The {component} {rng.choice(phrases)}." for component in rng.sample(components, count))
        rows.append({"id": i + 1, "review": text, "label": label})
    return rows


def read_jsonl(path: Path, text_field: str, max_chars: int | None) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL file does not exist: {path}")
    rows: list[dict] = []
    texts: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {lineno} must be a JSON object, not {type(row).__name__}")
            if text_field not in row:
                raise ValueError(
                    f"Line {lineno} is missing '{text_field}'. "
                    f"Use --text-field to select the review field; found {sorted(row)}"
                )
            text = row[text_field]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Line {lineno}: '{text_field}' must be a nonempty string")
            rows.append(row)
            texts.append(text[:max_chars] if max_chars is not None else text)
    if not rows:
        raise ValueError(f"Input JSONL file contains no valid records: {path}")
    return rows, texts


def normalized_label(answer: str) -> str | None:
    """Accept one-word labels, optionally followed by punctuation; reject explanations."""
    match = re.fullmatch(r"\s*(positive|negative)\s*[.!]?\s*", answer, re.IGNORECASE)
    return match.group(1).lower() if match else None


def mock_generate(texts: list[str]) -> tuple[list[str], int]:
    """Deterministic HEURISTIC ONLY: tests queueing and output, NOT LLM quality."""
    results = []
    for text in texts:
        lower = text.lower()
        neg = any(phrase in lower for phrase in NEGATIVE_PHRASES)
        pos = any(phrase in lower for phrase in POSITIVE_PHRASES)
        results.append("negative" if neg and not pos else "positive" if pos and not neg else "negative")
    return results, len(results)


def load_model(model_id: str, device: str, dtype: str | None):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install Transformers: python -m pip install transformers==5.22.2") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # decoder-only models generate after the final token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no EOS/padding token")
        tokenizer.pad_token = tokenizer.eos_token

    chosen_dtype = dtype or ("float16" if device.startswith("cuda") else "float32")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, chosen_dtype)
    )
    # Each worker explicitly owns exactly one device; no accelerate/device_map needed.
    model = model.to(torch.device(device)).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


@torch.inference_mode()
def generate_batch(model, tokenizer, texts: list[str], max_new_tokens: int, max_input_tokens: int):
    encoded = tokenizer.apply_chat_template(
        [build_messages(t) for t in texts],
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    output = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output[:, encoded["input_ids"].shape[1]:]
    answers = [answer.strip() for answer in tokenizer.batch_decode(generated, skip_special_tokens=True)]
    # Approximate generated tokens, including any first EOS but excluding subsequent pad tokens.
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    counts = 0
    for seq in generated.tolist():
        for token_id in seq:
            counts += 1
            if token_id in eos_ids:
                break
    return answers, counts


def worker(worker_id, device, args, threads, load_lock, task_queue, result_queue, go_event):
    """Top-level function is picklable under Windows multiprocessing spawn."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if device == "cpu":
            torch.set_num_threads(threads)
        else:
            torch.cuda.set_device(int(device.split(":", 1)[1]))

        if args.mock:
            model = tokenizer = None
        else:
            with load_lock:  # avoid concurrent weight downloads and RAM spikes
                tokenizer, model = load_model(args.model, device, args.dtype)

        name = torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU"
        result_queue.put(("ready", worker_id, f"{device}: {name}"))
        go_event.wait()
        while True:
            item = task_queue.get()
            if item is None:
                break
            ids, texts = item
            if args.mock:
                outputs, token_count = mock_generate(texts)
            else:
                outputs, token_count = generate_batch(
                    model, tokenizer, texts, args.max_new_tokens, args.max_input_tokens
                )
            result_queue.put(("batch", worker_id, (ids, outputs, token_count)))
        result_queue.put(("finished", worker_id, None))
    except BaseException:
        result_queue.put(("error", worker_id, traceback.format_exc()))


def receive(queue_, procs, timeout_seconds: int):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            kind, wid, payload = queue_.get(timeout=0.5)
        except queue.Empty:
            for proc in procs:
                if proc.exitcode is not None and proc.exitcode != 0:
                    raise RuntimeError(f"{proc.name} crashed with exit code {proc.exitcode}")
            if all(proc.exitcode is not None for proc in procs):
                raise RuntimeError("Workers exited before all results arrived; inspect worker errors")
            continue
        if kind == "error":
            raise RuntimeError(f"Worker {wid} failed:\n{payload}")
        return kind, wid, payload
    raise TimeoutError(f"No worker response within {timeout_seconds} seconds")


def checked_devices(requested: list[str] | None, mock: bool, dtype: str | None):
    count = torch.cuda.device_count()
    devices = requested or ([f"cuda:{i}" for i in range(count)] if count else ["cpu"])
    if len(devices) != len(set(devices)) and any(d.startswith("cuda:") for d in devices):
        raise ValueError("Duplicate CUDA devices are disabled: a second model replica can exhaust VRAM")
    for device in devices:
        if device == "cpu":
            if dtype in {"float16", "bfloat16"} and not mock:
                raise ValueError("Use float32 for CPU inference")
            continue
        if not re.fullmatch(r"cuda:\d+", device):
            raise ValueError(f"Invalid device '{device}'; expected 'cpu' or 'cuda:0', 'cuda:1', ...")
        idx = int(device.partition(":")[2])
        if idx >= count:
            raise ValueError(
                f"Requested {device}, but PyTorch detects {count} CUDA GPU(s). "
                "Check torch.cuda.is_available() and your PyTorch CUDA build."
            )
        if dtype == "bfloat16" and torch.cuda.get_device_capability(idx)[0] < 8:
            raise ValueError(f"{device} does not support efficient bfloat16; use float16")
    return devices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--devices", nargs="+", help="e.g. cuda:0 cuda:1 or cpu cpu for mock tests")
    parser.add_argument("--input", type=Path, help="UTF-8 JSONL input")
    parser.add_argument("--text-field", default="review", help="JSON field containing text (default: review)")
    parser.add_argument("--output", type=Path, default=Path("results.jsonl"))
    parser.add_argument("--num-samples", type=int, default=40, help="demo records when --input is omitted")
    parser.add_argument("--batch-size", type=int, default=2, help="examples per generate call")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--worker-timeout", type=int, default=900, help="timeout per worker response")
    parser.add_argument("--mock", action="store_true", help="offline heuristic backend: test multiprocessing only")
    args = parser.parse_args(argv)
    for key in ("batch_size", "max_new_tokens", "max_input_tokens", "worker_timeout"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("--max-chars must be positive")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.input:
        rows, texts = read_jsonl(args.input, args.text_field, args.max_chars)
    else:
        rows = make_demo_rows(args.num_samples)
        texts = [row["review"][:args.max_chars] for row in rows]
    devices = checked_devices(args.devices, args.mock, args.dtype)
    if not args.mock and any(d == "cpu" for d in devices):
        print("NOTE: CPU LLM inference can be very slow. Use --mock for a quick pipeline test.")
    if args.mock:
        print("MOCK MODE: heuristics only; no LLM loaded, no inference quality benchmark.")
    print(f"{len(rows)} rows -> {len(devices)} worker(s): {', '.join(devices)}", flush=True)

    order = sorted(range(len(texts)), key=lambda index: len(texts[index]), reverse=True)
    batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(4, len(devices) * 2))
    result_queue = ctx.Queue()
    go = ctx.Event()
    load_lock = ctx.Lock()
    threads = max(1, (os.cpu_count() or 1) // len(devices))
    procs = [
        ctx.Process(
            target=worker,
            name=f"worker-{i}({device})",
            args=(i, device, args, threads, load_lock, task_queue, result_queue, go),
            daemon=True,
        )
        for i, device in enumerate(devices)
    ]
    for proc in procs:
        proc.start()

    producer_error = []
    def feed_batches():
        try:
            for ids in batches:
                task_queue.put((ids, [texts[i] for i in ids]))
            for _ in devices:
                task_queue.put(None)
        except Exception as exc:
            producer_error.append(exc)

    answers: dict[int, str] = {}
    row_counts = [0 for _ in devices]
    token_count = 0
    try:
        load_start = time.perf_counter()
        ready = set()
        while len(ready) < len(devices):
            kind, wid, info = receive(result_queue, procs, args.worker_timeout)
            if kind != "ready" or wid in ready:
                raise RuntimeError(f"Unexpected initialization message: {kind} from {wid}")
            ready.add(wid)
            print(f"  Worker {wid} ready: {info}", flush=True)
        load_seconds = time.perf_counter() - load_start
        producer = threading.Thread(target=feed_batches, daemon=True)
        producer.start()
        go.set()
        run_start = time.perf_counter()
        with tqdm(total=len(rows), unit="row", desc="Inference") as progress:
            while len(answers) < len(rows):
                kind, wid, payload = receive(result_queue, procs, args.worker_timeout)
                if kind == "finished":
                    continue
                if kind != "batch":
                    raise RuntimeError(f"Unexpected work message: {kind}")
                ids, outputs, count = payload
                if len(ids) != len(outputs):
                    raise RuntimeError(f"Worker {wid} returned {len(outputs)} outputs for {len(ids)} inputs")
                if any(i in answers for i in ids):
                    raise RuntimeError(f"Worker {wid} returned a duplicate row index")
                answers.update(zip(ids, outputs, strict=True))
                row_counts[wid] += len(ids)
                token_count += count
                progress.update(len(ids))
        run_seconds = max(time.perf_counter() - run_start, 1e-9)
        producer.join(timeout=5)
        if producer_error:
            raise RuntimeError(f"Task producer failed: {producer_error[0]}")
        for proc in procs:
            proc.join(timeout=15)
        if any(proc.is_alive() for proc in procs):
            raise RuntimeError("Some inference workers did not shut down normally")
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)
        task_queue.cancel_join_thread()
        result_queue.cancel_join_thread()
        task_queue.close()
        result_queue.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for idx, row in enumerate(rows):
            answer = answers[idx]
            output_row = {**row, "llm_output": answer, "predicted_label": normalized_label(answer)}
            out.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print(f"Model load: {load_seconds:.2f}s | inference: {run_seconds:.2f}s | "
          f"{len(rows)/run_seconds:.2f} rows/s | ~{token_count/run_seconds:.0f} output tokens/s")
    for i, device in enumerate(devices):
        print(f"  Worker {i} ({device}) processed {row_counts[i]} rows")
    labeled = [i for i, row in enumerate(rows) if row.get("label") in ("positive", "negative")]
    if labeled:
        good = sum(normalized_label(answers[i]) == rows[i]["label"] for i in labeled)
        print(f"Sentiment accuracy: {good}/{len(labeled)} ({good/len(labeled):.1%})"
              + (" [MOCK; not an LLM benchmark]" if args.mock else ""))
    print(f"Output file: {args.output.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()  # recommended for Windows executables and multiprocessing
    main()
