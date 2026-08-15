# LogSense — Project Details

A complete technical write-up of what LogSense is, what it builds on, how it
was trained, and how the pieces fit together. For setup/run instructions see
[README.md](README.md).

## 1. Problem

HDFS (Hadoop Distributed File System) clusters produce huge volumes of
semi-structured logs. Manually spotting the sequences that indicate a real
failure (a lost block, a replication timeout, a mirrored write exception) in
millions of log lines is impractical. Automated anomaly detection over log
sequences is a well-studied problem — but most systems stop at a binary
"anomalous / not anomalous" label, which still leaves an engineer to work out
*why* the sequence was flagged.

LogSense adds that missing second step: when a block is flagged anomalous,
it automatically generates a human-readable explanation of what went wrong.

## 2. Base architecture: LogLLM

The classifier stage builds directly on **LogLLM** (Guan et al., 2025,
[arXiv:2411.08561](https://arxiv.org/abs/2411.08561) — the paper in
`docs/research-paper/`). LogLLM frames log anomaly detection as a
generation task instead of classification-head fine-tuning:

- **BERT** (`bert-base-uncased`) encodes each log message; the `[CLS]` token
  output goes through a linear layer to produce a semantic vector per line.
- A **projector** (a single linear layer) maps that vector from BERT's
  embedding space into Llama's token-embedding space, so log-line embeddings
  can be spliced directly into a Llama prompt in place of text tokens.
- **Llama-3-8B** consumes `[intro tokens] + [projected log embeddings] +
  "Is this sequence normal or anomalous?"` and generates `"The sequence is
  normal/anomalous."` — classification framed as next-token generation.
- No traditional log parser (e.g. Drain) is needed: raw log text is
  regex-masked to replace variable fields (IPs, paths, block IDs, etc.) with
  a `<*>` wildcard, and messages are grouped into sequences by **session
  window** (HDFS: grouped by `block_id`).
- Training uses **QLoRA** (4-bit) across three stages: (1) fine-tune Llama
  alone on the answer template, (2) train the BERT+projector embedder to
  align with Llama's space, (3) jointly fine-tune everything with QLoRA.
  Minority-class oversampling handles the natural class imbalance (HDFS is
  ~2.9% anomalous).
- On HDFS, the paper reports precision 0.994 / recall 1.000 / F1 0.997,
  outperforming 8 baselines (DeepLog, LogAnomaly, LogBERT, etc). Official
  upstream code: [guanwei49/LogLLM](https://github.com/guanwei49/LogLLM),
  vendored unmodified in `reference/LogLLM-master/` for comparison.

`backend/classifier/model.py` and `custom_dataset.py` are this project's
working copy of the LogLLM classifier — same architecture, adapted to load
Llama with an explicit Hugging Face token (needed since Llama 3 is gated).

## 3. This project's extension: the explainer stage

LogLLM's own output is one sentence: "the sequence is anomalous." LogSense
adds a **second model, triggered only on a positive classification**, that
explains *why*:

```
Stage 1 (classifier)              Stage 2 (explainer, conditional)
─────────────────────             ──────────────────────────────────
BERT + Llama-3-8B (QLoRA)    →    Llama-3.2-3B-Instruct + LoRA
trained on HDFS sessions          trained on a custom synthetic dataset
outputs: normal / anomalous       outputs: 3-5 sentence root-cause explanation
```

- **Base model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Adapter**: [`kovidritesh/Llama-3.2-3B-FineTuned`](https://huggingface.co/kovidritesh/Llama-3.2-3B-FineTuned)
  — the LoRA adapter trained for this project (see §4), published to Hugging
  Face Hub and pulled at inference time by `backend/app.py` and
  `backend/explainer/inference.py` (default value of `EXPLAINER_ADAPTER_REPO`
  in `backend/config.py` / `.env.example`).
- Fine-tuned with **Unsloth + LoRA (r=16)** via `trl.SFTTrainer`, ~25 minutes
  on an A100, 4-bit quantized for inference — see
  `notebooks/03_explainer_model_training.ipynb`.

### Why a second model instead of one bigger classifier?

Splitting classification and explanation into two models keeps the
expensive 8B classifier focused on what it's good at (accurate binary
detection over long sequences) while the 3B explainer — only invoked for the
~3% of blocks that are actually anomalous — handles free-form generation.
This keeps average inference cost low (most blocks never reach Stage 2)
while still producing rich, human-readable diagnostics for the anomalies
that matter.

## 4. Building the explainer's training dataset

The explainer needed a dataset it doesn't get from LogLLM or Loghub: pairs of
`(anomalous block details) → (natural-language root-cause explanation)`.
None exists publicly, so this project generated one synthetically:

1. **Source data**: `Event_traces.csv` from the preprocessed HDFS_v1 Loghub
   dataset — one row per block, with its `Type` (anomaly category, or NaN
   for normal), event-ID sequence, and latency.
2. **Filtering** (`backend/data_pipeline/generate_explanations_llm.py`):
   keep only anomalous blocks (`Type` not NaN), drop sequences that are too
   short (<3 events) or absurdly long (>60), and deduplicate by sequence
   hash so the LLM isn't asked to explain the same pattern hundreds of times.
3. **Generation**: for each remaining block, an LLM (this project used
   Groq/OpenRouter-hosted Llama-3.3-70B during development, configurable via
   `.env`) is prompted with the block's event sequence translated into plain
   English (via `HDFS_log_templates.csv`), its anomaly type, and its
   latency, and asked to write a 3–5 sentence causal explanation —
   explicitly instructed not to use raw event codes (`E5`, `E22`, etc.) and
   not to write a generic, sequence-agnostic answer.
4. **Quality filtering**: generated explanations are rejected if they're too
   short, start with a filler phrase ("Here is...", "Certainly..."), or have
   fewer than two real sentences — rejected/accepted samples are logged
   separately so a crashed or rate-limited run can resume without
   re-generating already-accepted examples.
5. **Assembly** (`notebooks/01_dataset_preparation.ipynb`): the generated
   explanations are converted into OpenAI-style chat records
   (`{system, user, assistant}` messages) and combined/deduplicated into the
   final training/validation split:
   `data/explainer_dataset/logsense_final_train.jsonl` (565 examples) /
   `logsense_final_val.jsonl` (63 examples).

A separate, deterministic fallback generator
(`backend/data_pipeline/generate_explanations_rule_based.py`) exists too —
it builds explanations from a fixed event-phrase dictionary and a rule set
(no LLM calls, no API cost), useful if you want to regenerate data without
depending on an external API.

## 5. Serving: two backend modes

- **`backend/app.py`** — the real pipeline. Loads both fine-tuned models
  into GPU memory and serves `POST /analyze` (classify, then explain if
  anomalous) and `GET /live-log` (streams the sample HDFS log for the
  frontend's live-monitor panel). This is what actually runs in production
  and is the architecturally "correct" version, but it needs a capable GPU
  (developed on Kaggle's dual T4s: classifier on one, explainer on the
  other, both 4-bit quantized).
- **`backend/app_lite.py`** — a lighter alternative with the *identical*
  request/response contract, for environments without a GPU (a laptop, a
  grading environment, a quick demo). Instead of running the fine-tuned
  models, it sends one prompt to a hosted LLM (Gemini by default, or any
  OpenAI-compatible API) asking it to both classify and explain in a single
  call, returning the same JSON shape the frontend expects. This is the
  version that was actually used for live demos of this project, since it
  starts in seconds and needs no GPU or model downloads.

Both are configured entirely through environment variables (`.env`, see
`.env.example`) — no secrets or paths are hardcoded in either file.

## 6. Frontend

`frontend/` is a React + Vite dashboard: a form to submit a block's event
sequence and latency, sent to `POST {API_BASE}/analyze`; the result (status +
explanation) is displayed alongside dashboard widgets (stat cards, an
anomaly-distribution chart, a live-log panel polling `GET /live-log`, and a
history table of past analyses). `API_BASE` is read from
`VITE_API_BASE_URL` (`frontend/.env`) so it can point at either backend mode
without code changes.

## 7. Repository history note

This repo was reorganized from a working-but-messy Kaggle-notebook-driven
project into the structure described in `README.md`. During that cleanup:

- Several hardcoded API keys/tokens (Hugging Face, OpenRouter, Gemini,
  ngrok) that existed in earlier working copies were removed from all
  tracked files and replaced with environment variables.
- Duplicate/superseded files (old frontend prototypes, a duplicate dataset
  generator script, an earlier Gemini-only backend prototype, a duplicate
  Kaggle notebook) were identified and set aside rather than deleted, in
  case anything was still needed.
- The upstream LogLLM repository is kept unmodified in `reference/` for
  direct comparison against this project's adapted copy in
  `backend/classifier/`.
