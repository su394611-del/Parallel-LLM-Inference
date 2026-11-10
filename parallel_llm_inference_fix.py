
"""Parallel LLM data processing across several GPUs.

Pattern: one copy of the model per GPU, one worker process per copy, and one
shared queue of batches. Whichever GPU is free pulls the next batch, so a
faster card automatically does more of the work.

    parent process                           worker process (one per GPU)
    --------------                           ----------------------------
    read rows, sort by length, cut           load its own copy of the model
    into batches, put them on a queue  --->  pull a batch, run generate()
    collect answers, restore the       <---  send (row ids, answers) back
    original order, write JSONL

Examples:
    python parallel_llm_inference.py                    # every visible GPU, built-in demo data
    python parallel_llm_inference.py --devices cuda:0   # one GPU only, to measure the speed-up
    python parallel_llm_inference.py --input reviews.jsonl --text-field review --output out.jsonl

Requires: Python 3.10+, torch, transformers>=4.56, accelerate
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import signal
import threading
import time
import traceback

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_messages(text):
    """The task. Edit this prompt to summarise, extract, translate, ... instead."""
    return [{
        "role": "user",
        "content": "Classify the sentiment of this product review as positive or negative. "
                   f"Answer with one word.\n\nReview: {text}",
    }]


def demo_rows(n, seed=0):
    """Synthetic reviews of varying length whose true sentiment is known, so the answers can be checked."""
    rng = random.Random(seed)
    parts = ["battery", "screen", "keyboard", "camera", "speaker", "charger", "touchpad", "microphone", "fan", "hinge"]
    opinions = {
        "positive": ["is fantastic", "exceeded my expectations", "works flawlessly",
                     "is the best I have ever used", "was a pleasant surprise"],
        "negative": ["is terrible", "was a huge disappointment", "stopped working after a week",
                     "is far worse than advertised", "keeps failing"],
    }
    rows = []
    for _ in range(n):
        label = rng.choice(["positive", "negative"])
        chosen = rng.sample(parts, rng.randint(1, 6))
        rows.append({"text": " ".join(f"The {p} {rng.choice(opinions[label])}." for p in chosen), "label": label})
    return rows


def load_model(name, device, dtype):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # the model continues from the last token, so padding must go on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if dtype is None:  # half precision on a GPU: half the memory and faster
        dtype = "float16" if device.startswith("cuda") else "float32"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), device_map={"": device})
    return tok, model.eval()


@torch.inference_mode()
def generate(model, tok, texts, max_new_tokens):
    """Run one batch. Returns the answers and the number of tokens generated."""
    enc = tok.apply_chat_template(
        [build_messages(t) for t in texts],
        add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[:, enc["input_ids"].shape[1]:]  # drop the prompt, keep only the answer
    answers = tok.batch_decode(new_tokens, skip_special_tokens=True)
    return [a.strip() for a in answers], int((new_tokens != tok.pad_token_id).sum())


def exit_with_parent():
    """Watchdog thread: if the parent is killed, do not stay behind holding GPU memory."""
    mp.parent_process().join()
    os._exit(1)


def worker(wid, device, args, n_threads, load_lock, tasks, results, go):
    """Runs in its own process: owns one model copy on one device and drains the shared queue."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the parent, which then stops the workers
    threading.Thread(target=exit_with_parent, daemon=True).start()
    try:
        if device == "cpu":
            torch.set_num_threads(n_threads)  # keep CPU workers from fighting over the same cores
        with load_lock:  # load one copy at a time: a single download and a lower peak in system RAM
            tok, model = load_model(args.model, device, args.dtype)
        gpu_name = f" ({torch.cuda.get_device_name(device)})" if device.startswith("cuda") else ""
        results.put(("ready", wid, device + gpu_name))
        go.wait()
        while (batch := tasks.get()) is not None:  # None means "no more work"
            ids, texts = batch
            answers, n_tokens = generate(model, tok, texts, args.max_new_tokens)
            results.put(("batch", wid, (ids, answers, n_tokens)))
    except Exception:
        results.put(("error", wid, traceback.format_exc()))


def next_message(results, procs):
    """Next message from any worker. Raises instead of waiting forever if a worker failed or was killed."""
    while True:
        for p in procs:  # checked on every message, so a dead worker is noticed while the others keep going
            if p.exitcode not in (None, 0):
                raise RuntimeError(f"{p.name} died with exit code {p.exitcode} (killed - out of memory?)")
        try:
            kind, wid, payload = results.get(timeout=1)
        except queue.Empty:
            continue
        if kind == "error":
            raise RuntimeError(f"worker {wid} failed:\n{payload}")
        return kind, wid, payload


def main():
    args = parse_args()
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = demo_rows(args.num_samples)
    texts = [str(row[args.text_field])[:args.max_chars] for row in rows]

    devices = args.devices or [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    print(f"{len(rows)} rows -> {len(devices)} worker(s) on {' '.join(devices)}")

    # Put rows of similar length in the same batch (less padding = less wasted compute). Longest
    # first, so a batch that is too big for the GPU fails in the first seconds, not at the end.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]

    ctx = mp.get_context("spawn")  # each worker must be a fresh process: CUDA does not survive a fork
    tasks, results, go, load_lock = ctx.Queue(), ctx.Queue(), ctx.Event(), ctx.Lock()
    for ids in batches:
        tasks.put((ids, [texts[i] for i in ids]))
    for _ in devices:
        tasks.put(None)  # one "no more work" marker per worker
    n_threads = max(1, (os.cpu_count() or 1) // len(devices))
    procs = [
        ctx.Process(target=worker, name=f"worker {wid} ({device})", daemon=True,
                    args=(wid, device, args, n_threads, load_lock, tasks, results, go))
        for wid, device in enumerate(devices)
    ]
    for p in procs:
        p.start()

    answers, rows_done, n_tokens = {}, [0] * len(devices), 0
    try:
        t0 = time.perf_counter()
        for _ in devices:  # wait until every copy of the model is loaded ...
            _, wid, where = next_message(results, procs)
            print(f"  worker {wid} ready on {where}")
        load_s = time.perf_counter() - t0
        go.set()  # ... then start them together, so the timing below is inference only
        t0 = time.perf_counter()
        with tqdm(total=len(rows), unit="row") as bar:
            while len(answers) < len(rows):
                _, wid, (ids, outs, n) = next_message(results, procs)
                answers.update(zip(ids, outs, strict=True))
                rows_done[wid] += len(ids)
                n_tokens += n
                bar.update(len(ids))
        run_s = time.perf_counter() - t0
    except BaseException:
        for p in procs:
            p.terminate()
        tasks.cancel_join_thread()  # do not wait to hand out work nobody will take
        raise
    for p in procs:
        p.join()

    with open(args.output, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):  # original order, whatever order the GPUs finished in
            f.write(json.dumps({**row, "llm_output": answers[i]}, ensure_ascii=False) + "\n")

    print(f"Model load {load_s:.1f}s | inference {run_s:.1f}s = "
          f"{len(rows) / run_s:.1f} rows/s, {n_tokens / run_s:.0f} generated tokens/s")
    for wid, device in enumerate(devices):
        print(f"  worker {wid} on {device}: {rows_done[wid]} rows")
    if not args.input and rows:  # the demo rows carry their true label: right = names it and not the opposite
        said = [{word for word in ("positive", "negative") if word in answers[i].lower()} for i in range(len(rows))]
        correct = sum(said[i] == {row["label"]} for i, row in enumerate(rows))
        print(f"Accuracy on the demo labels: {correct / len(rows):.1%}")
    print(f"Wrote {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="a chat model on the Hugging Face Hub, or a local folder")
    p.add_argument("--devices", nargs="+",
                   help="e.g. cuda:0 cuda:1 (default: every visible GPU, else cpu); repeat a device to put two copies on it")
    p.add_argument("--input", help="JSONL file, one JSON object per line (default: built-in demo reviews)")
    p.add_argument("--text-field", default="text", help="field of each row that is sent to the model")
    p.add_argument("--output", default="results.jsonl")
    p.add_argument("--num-samples", type=int, default=1000, help="number of demo rows")
    p.add_argument("--batch-size", type=int, default=32, help="rows per generate() call; lower it if a GPU runs out of memory")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-chars", type=int, help="cut longer inputs to this many characters (default: no limit)")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], help="default: float16 on GPU, float32 on CPU")
    return p.parse_args()


if __name__ == "__main__":  # required: every worker process re-imports this file
    main()
