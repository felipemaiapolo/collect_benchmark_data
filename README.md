# Collect Benchmark Data

This repository provides a lightweight pipeline for gathering model responses on a set of popular reasoning and instruction-following benchmarks. It orchestrates prompt construction, model querying across multiple providers, optional judging with an OpenAI model, and JSON/JSONL utilities for storing results.

## Supported benchmarks

The `run` helper in `run.py` can be pointed at any of the benchmarks below. Each benchmark module contains small helpers for formatting prompts or extracting references from the source dataset.

| Benchmark key | Dataset source | Helper module |
| --- | --- | --- |
| `ifeval` | [`google/IFEval`](https://huggingface.co/datasets/google/IFEval) | `ifeval.utils.process_results` |
| `math` | [`DigitalLearningGmbH/MATH-lighteval`](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) (Level 5) | `mathbench.utils.doc_to_text` |
| `musr` | [`TAUR-Lab/MuSR`](https://huggingface.co/datasets/TAUR-Lab/MuSR) | `musr.utils.doc_to_text` |
| `gpqa` | [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa) (`gpqa_main`) | `gpqa.utils.process_docs` / `gpqa.utils.doc_to_text` |
| `bbh` | [`SaylorTwift/bbh`](https://huggingface.co/datasets/SaylorTwift/bbh) | `bbh.utils.doc_to_text` |
| `mmlu-pro` | [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | `mmlupro.utils.doc_to_text` / `mmlupro.utils.doc_to_choice` |

## Installation

1. Use Python 3.10+ and create a virtual environment if desired.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The pipeline uses the Hugging Face `datasets` library to fetch benchmark data at runtime, so network access is required for the first download.

## Configure API access

Model and judge calls rely on API keys defined in `utils.py`. Populate the following constants before running evaluations:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `TOGETHER_API_KEY`
- `MISTRAL_API_KEY`
- `HF_TOKEN`
- `RITS_API_KEY`

Each key feeds the corresponding provider block in `QueryModel` (e.g., OpenAI, Anthropic, Google Gemini, Mistral, TogetherAI, Hugging Face Inference Endpoints, and RITS). The judge model defaults to `gpt-4o-mini-2024-07-18` inside `QueryJudgeError`.

## Running an evaluation

The entry point is the `run` function in `run.py`:

```python
from run import run

# Evaluate 50 items from the BBH benchmark with an OpenAI model
run("bbh", model="gpt-4o-2024-05-13", n=50)
```

Arguments:

- `benchmark`: one of the keys listed in the table above.
- `model`: any model name present in `utils.MODELS` (the provider is auto-detected by `GetAPI`). Slashes and underscores are normalized when building result paths.
- `n` (optional): number of examples to sample uniformly from the benchmark (defaults to 200).

### What happens during `run`

1. The benchmark dataset is loaded via `datasets.load_dataset` and converted into a list of documents.
2. A set of evenly spaced example IDs is generated from the corpus (`n` points across the dataset length).
3. For each ID, the script constructs a prompt and expected answer using the benchmark helper module.
4. The target model is queried with `QueryModelError`, which retries transient failures up to `max_tries` and records `"NA"` if unsuccessful.
5. If a model response is available, it is scored by `QueryJudgeError` (except for `ifeval`, which uses the benchmark’s native scoring).
6. Results are saved as JSON in `results/<benchmark>/<model>/`.

## Output layout

Each evaluation produces one JSON file per example with the following fields:

```json
{
  "model": "gpt-4o-2024-05-13",
  "benchmark": "bbh",
  "id": 17,
  "id_bench": "",
  "prompt": "Question: ...\n\nAnswer:",
  "model_response": "...",
  "gold_response": "...",
  "subject": "logical_deduction_three_objects",
  "scores": { "correctness_score": 1.0, "justification": "..." },
  "judge_prompt": "...",
  "doc": { ...raw benchmark entry... }
}
```

Existing files are re-used: if `model_response` or `scores` are recorded as `"NA"`, the script will attempt to fill them in on subsequent runs before moving to new examples.

## Repository structure

- `run.py` – orchestrates dataset loading, prompting, model calls, and judging.
- `utils.py` – provider registry, API wrappers, retry helpers, and JSON/JSONL utilities.
- `bbh/`, `gpqa/`, `mathbench/`, `mmlupro/`, `musr/`, `ifeval/` – per-benchmark formatting helpers.
- `collect_data.ipynb` – exploratory notebook for interactive data collection.

## Troubleshooting

- **Authentication errors**: double-check that the provider keys in `utils.py` are set and valid. Some providers (e.g., Hugging Face Inference Client) require specific tokens per hosting provider listed in `utils.MODELS`.
- **Rate limits or transient failures**: `QueryModelError` and `QueryJudgeError` automatically retry with a short delay. Increase `max_tries` in those helpers if needed.
- **Partial runs**: Because outputs are written per example, you can interrupt and resume runs without losing completed evaluations.
