# LogSense

LogSense is a two-stage log anomaly detection system for HDFS logs. It builds
on [LogLLM](https://arxiv.org/abs/2411.08561) (BERT + Llama-3-8B) as an
anomaly **classifier**, and adds a second stage: a LoRA-fine-tuned
**Llama-3.2-3B explainer** that generates a plain-English root-cause
explanation whenever the classifier flags a block as anomalous.

```
HDFS log → parse into per-block event sequences
         → Stage 1: LogLLM classifier  → normal / anomalous?
                                            │ if anomalous
                                            ▼
         → Stage 2: Llama-3.2-3B explainer → "here's what went wrong"
```

A React dashboard (`frontend/`) visualizes results in real time, and a
FastAPI backend (`backend/`) serves both stages behind `/analyze` and
`/live-log` endpoints.

## Repository layout

```
backend/            FastAPI server + the ML pipeline
  app.py               full pipeline: real LogLLM classifier + Llama explainer (needs a GPU)
  app_lite.py          API-key alternative: one hosted LLM call, no GPU needed (for demos)
  classifier/          LogLLM model definition (BERT + Llama-3-8B, QLoRA)
  explainer/           explainer inference (Llama-3.2-3B + LoRA)
  data_pipeline/        log parsing + explainer dataset generation

frontend/            React + Vite dashboard (talks to the backend's /analyze, /live-log)

notebooks/           the Kaggle notebooks used to build the dataset, train, and demo the pipeline

data/                sample dataset (committed) + full dataset (gitignored, see data/README.md)

models/              fine-tuned model configs (weights gitignored, see models/README.md)

docs/                research paper, project report/slides, design docs, screenshots
```

`reference/LogLLM-master/` (the unmodified upstream [LogLLM](https://github.com/guanwei49/LogLLM)
repo this project's classifier builds on) isn't vendored in this repo — pull
it yourself if you want to diff against it:

```bash
git clone https://github.com/guanwei49/LogLLM.git reference/LogLLM-master
```

## Quickstart

### 1. Full pipeline (real models, needs a GPU)

```bash
cp .env.example .env        # fill in HF_TOKEN at minimum
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

`HF_TOKEN` needs read access to `meta-llama/Meta-Llama-3-8B` and
`meta-llama/Llama-3.2-3B-Instruct` (accept Meta's license for both on
Hugging Face first). This mode was developed/run on Kaggle's dual-T4 GPU
runtime — see `models/README.md` for what weights it expects locally.

If you're running from a host with no public IP (Kaggle/Colab notebook),
set `NGROK_AUTHTOKEN` and run `python -m backend.app` instead, which tunnels
the server through ngrok and prints a public URL.

### 2. Lite mode (no GPU — for demos)

```bash
cp .env.example .env        # set LLM_PROVIDER=gemini and GEMINI_API_KEY
pip install -r backend/requirements.txt
uvicorn backend.app_lite:app --host 0.0.0.0 --port 8000
```

Same `/analyze` / `/live-log` contract as the full pipeline, but delegates
classification + explanation to a single hosted LLM call (Gemini by default;
any OpenAI-compatible provider works too — see `.env.example`). This is what
was actually used for live demos of this project.

### 3. Frontend

```bash
cd frontend
cp .env.example .env        # point VITE_API_BASE_URL at whichever backend you ran above
npm install
npm run dev
```

## Notebooks

The `notebooks/` directory holds the original Kaggle notebooks this project
was developed in — useful for seeing the full training/eval process and
outputs, but `backend/` is the cleaned-up, structured version of the same
code and is what you should actually run or extend.

| Notebook | Purpose |
|---|---|
| `01_dataset_preparation.ipynb` | Builds the explainer's fine-tuning dataset (JSONL) from HDFS event traces + generated explanations |
| `02_classifier_explainer_pipeline_demo.ipynb` | End-to-end run: parse → classify → explain, on the full HDFS log |
| `03_explainer_model_training.ipynb` | Fine-tunes Llama-3.2-3B-Instruct with Unsloth + LoRA on the explainer dataset |
| `04_backend_api_kaggle_server.ipynb` | The original Kaggle-hosted FastAPI + ngrok server (now superseded by `backend/app.py`) |

## Security note

Earlier working copies of this project had API keys and Hugging Face tokens
hardcoded directly in scripts and notebooks. Those have all been removed and
replaced with environment variables (`.env`, gitignored) as part of
preparing this repo for GitHub. **If you're the original author: the leaked
keys (Hugging Face — including one with write access, OpenRouter, Gemini,
ngrok) should be rotated/revoked regardless, since they were exposed before
this cleanup.**

## License

MIT — see [LICENSE](LICENSE). The classifier architecture builds on
[LogLLM](https://github.com/guanwei49/LogLLM) (Guan et al., 2025), also MIT.
