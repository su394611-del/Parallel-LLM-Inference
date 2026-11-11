# Parallel LLM Inference with PyTorch, Hugging Face, and Multi-GPU Workers

An  **batch inference** project for classifying product-review sentiment using **Qwen2.5-0.5B-Instruct** and a multiprocessing worker pool. Each worker loads a **separate model replica** on its assigned GPU. Workers pull batches from a shared queue; the parent restores input order and writes predictions to JSONL.

> **Scope:** This is parallel **inference**, not DDP training, model/tensor parallelism, or an HTTP-serving API. Each GPU must have enough VRAM for the entire model and its generation buffers. The included `--mock` backend is a deterministic keyword heuristic, **not** an LLM and **not** an accuracy benchmark.

## Features

- Parallel inference using spawned worker processes (`multiprocessing`), suitable for Windows and notebook-launched subprocesses.
- Single GPU, multiple independent GPUs, or multiple CPU workers for offline mock tests.
- Dynamic task queue and length-sorted mini-batches to reduce padding overhead.
- Hugging Face chat-template prompts, left-padding, greedy generation, and configurable input/output token caps.
- Robust JSONL validation and stable output order, with `llm_output` and normalized `predicted_label` fields.
- Separate model-initialization and processing timings; per-worker record counts and optional labeled-example accuracy.
- Includes a **self-contained Colab/Kaggle notebook** and synthetic dataset to reproduce the workflow.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── parallel_llm_inference.py       # Main, corrected Python CLI
├── Parallel_LLM_Inference_Colab_Kaggle.ipynb
├── reviews.jsonl                   # 48 synthetic, labeled input records
├── .gitignore
└── legacy/                         # Uploaded earlier variants, for reference
    ├── parallel_llm_inference_corrected_kaggle_original.py
    ├── parallel_llm_inference_fix_original.py
    └── parallel_llm_inference_colab_original.ipynb
```

**Recommended entry point:** `parallel_llm_inference.py`. The reference files in `legacy/` are not used by the instructions below. The previously supplied file `Parallel_llm_inference_requirements.txt` contained program text, not installable requirements, and is deliberately not treated as a dependency list.

## Requirements

- Python **3.10+** (Python 3.12 recommended for Windows).
- PyTorch (`torch`), `tqdm`, `transformers`, `safetensors` as listed in [`requirements.txt`](requirements.txt).
- Internet access and Hugging Face model download for real Qwen inference (unless already cached).
- Sufficient **free memory on each GPU** for a Qwen model replica, KV cache, and input batch. A 4 GB consumer GPU may work with short inputs and `--batch-size 1`, but out-of-memory errors are possible.
- Optional Jupyter installation for the notebook. **No CUDA GPU or model download is required for `--mock`.**

### Installation — Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

For a GPU, select and install the matching **CUDA-enabled PyTorch wheel** from [PyTorch's official installer](https://pytorch.org/get-started/locally/) first. The GPU driver version displayed by `nvidia-smi` is *not* the same as the CUDA runtime included with PyTorch. Check the resulting installation:

```powershell
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA usable:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count())"
```

If PyTorch reports `+cpu` or `torch.version.cuda` is `None`, the environment does **not** have CUDA-enabled PyTorch, even when `nvidia-smi` detects your NVIDIA GPU. Avoid replacing a working GPU build inadvertently when installing other dependencies.

### Installation — Colab / Kaggle

In a notebook cell (if packages are absent):

```python
%pip install "tqdm>=4.66" "transformers==5.22.2" "safetensors>=0.4"
```

Colab/Kaggle generally provide PyTorch already; verify the accelerator before modifying its installation. Select a GPU-enabled runtime in Colab; where available, Kaggle's **two T4 GPUs** configuration allows testing actual two-GPU inference. Standard Colab runtimes usually expose just one GPU.

## Quick start: fully offline test

Runs **two CPU worker processes** and the mock heuristic. This tests queueing, JSONL validation, batch distribution, and output writing; it does not load Qwen.

```powershell
python parallel_llm_inference.py --mock --devices cpu cpu --input reviews.jsonl --output results_mock.jsonl --batch-size 4
```

Or generate built-in synthetic rows (no input file needed):

```powershell
python parallel_llm_inference.py --mock --devices cpu cpu --num-samples 20 --output results_mock.jsonl
```

## Real inference: Qwen2.5-0.5B-Instruct

**One CUDA GPU** (start here if your GPU has 4 GB VRAM):

```powershell
python parallel_llm_inference.py --devices cuda:0 --input reviews.jsonl --output results_gpu.jsonl --batch-size 1 --max-input-tokens 192 --max-new-tokens 8
```

**Two CUDA GPUs** (two *independent replicas*):

```bash
python parallel_llm_inference.py --devices cuda:0 cuda:1 --input reviews.jsonl --output results_2gpu.jsonl --batch-size 2 --max-input-tokens 192 --max-new-tokens 8
```

**Real CPU inference** is possible but can be slow:

```powershell
python parallel_llm_inference.py --devices cpu --input reviews.jsonl --output results_cpu.jsonl --batch-size 1
```

Do **not** specify the same GPU twice as separate `--devices` values; the corrected CLI rejects duplicate CUDA devices to avoid duplicate model copies exhausting VRAM.

## Run the Jupyter notebook

Open [`Parallel_LLM_Inference_Colab_Kaggle.ipynb`](Parallel_LLM_Inference_Colab_Kaggle.ipynb) in VS Code, Colab, Kaggle, or JupyterLab. The notebook is **self-contained**: a cell writes the full Python inference script to disk and then executes it through `subprocess`. This is deliberate—Windows `spawn` workers must import a real Python file, and direct multiprocessing execution inside `ipykernel` is unreliable.

Use **Run All** to exercise the offline mock path. Real Hugging Face inference is **opt-in**, controlled by `RUN_REAL_INFERENCE = False` in the notebook. The notebook also shows how to launch single- and multi-GPU runs if available. Do not run the script body with `%run` or paste its worker function into a notebook cell when using multiprocessing.

## Input JSONL format

Each nonempty line must be a separate JSON object with a `review` field containing a nonempty string:

```jsonl
{"id": 1, "review": "The display is excellent.", "label": "positive"}
{"id": 2, "review": "The battery is terrible.", "label": "negative"}
```

The optional `label` can be used for classification accuracy. To use a different input key, pass `--text-field text` (or the appropriate field name). Invalid JSON and missing or empty text fields produce a clear error.

Output `results_*.jsonl` preserves all original fields and adds:

```jsonl
{"id": 1, "review": "The display is excellent.", "label": "positive", "llm_output": "positive", "predicted_label": "positive"}
```

`predicted_label` is set to `null` if model output does not match the expected one-word format. Accuracy is computed only for rows with `label` values `positive` or `negative`.

## Common CLI options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model identifier / local folder |
| `--devices` | All visible CUDA GPUs, else `cpu` | Worker devices, e.g. `cuda:0 cuda:1` or `cpu cpu` |
| `--input` | None | JSONL input file (without it, synthetic demo rows are created) |
| `--text-field` | `review` | Text key in each JSONL record |
| `--output` | `results.jsonl` | Output JSONL destination |
| `--num-samples` | 40 | Number of generated rows with no input file |
| `--batch-size` | 2 | Rows per `generate()` call per worker |
| `--max-input-tokens` | 256 | Tokenization limit for prompt plus review |
| `--max-new-tokens` | 8 | Maximum response tokens |
| `--dtype` | GPU: `float16`, CPU: `float32` | Optional explicit dtype |
| `--worker-timeout` | 900 | Seconds to wait for each worker response |
| `--mock` | Off | Skip model loading and test the queue with a heuristic |

For exact supported flags, run `python parallel_llm_inference.py --help`.

## Architecture

```text
 reviews.jsonl / generated data
               |
           Parent process
      validates records + sorts by length
               |
           Shared task queue
             /       \
       GPU 0 worker  GPU 1 worker
       model copy     model copy
             \       /
           Result queue
               |
    reorder by original row index
               |
       results_2gpu.jsonl
```

The implementation measures inference processing separately from startup/model loading. Speedup is **not** guaranteed—short jobs may be dominated by model loading, disk access, tokenization, or shared hardware constraints. Each worker stores a full model replica; **VRAM is not pooled**, and this is neither vLLM nor an always-on API service.

## Troubleshooting

- **`ModuleNotFoundError`**: install with `python -m pip install -r requirements.txt` in the *same Python environment* used to execute the script.
- **`CUDA available: False`**: use the official CUDA PyTorch build in a suitable Python environment; the NVIDIA driver alone is insufficient.
- **Out-of-memory**: use `--batch-size 1`, shorten `--max-input-tokens`, or switch to a smaller model / more VRAM. Stop GPU-heavy background processes.
- **Notebook multiprocessing errors**: launch the `.py` file in a subprocess; the supplied notebook already does this.
- **Worker initialization timeout**: the first Hugging Face model download can take time; check network/cache and consider increasing `--worker-timeout`.
- **Unexpected or blank model output**: inspect `llm_output` rather than assuming `predicted_label` will always be populated; greedy output may not follow the one-word prompt.
- **Windows PowerShell**: run Python snippets with `python -c "..."` or in a Python interpreter; don't paste `from ... import ...` directly at the `PS>` prompt.

## Validation and limitations

- Offline `--mock` verifies multiprocessing and file I/O but **cannot measure LLM classification accuracy**.
- The example dataset is synthetic and intentionally easy; results are not suitable for evaluating model generalization.
- Real Qwen inference and physical two-GPU scaling require downloading the model and having compatible compute; do not infer their performance from mock results.
- No HTTP endpoint, database persistence, distributed model partitioning, authentication, or production monitoring is included.
- Dependencies may change. Use pinned/compatible versions and create a fresh virtual environment for reproducible experiments.

## References

- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
- [PyTorch CUDA availability](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html)
- [Hugging Face Transformers text generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

*Educational code; review model licenses, dataset privacy, dependencies, and environment requirements before reuse.*
