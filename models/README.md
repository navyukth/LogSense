# Models

This project fine-tunes two models. Their **configs** are committed to git;
their **weights** (`.safetensors` / `.pt`) are gitignored — they're large
binary artifacts that don't belong in git history. Get them from Hugging
Face Hub or by retraining with the notebooks in `notebooks/`.

## `classifier/ft_model_HDFS/` — LogLLM anomaly classifier

BERT + Llama-3-8B (see `backend/classifier/model.py`), QLoRA-fine-tuned on
HDFS logs to classify a block's event sequence as `normal` / `anomalous`.

- `Bert_ft/` — LoRA adapter for the BERT encoder
- `Llama_ft/` — LoRA adapter for the Llama-3-8B decoder
- `projector.pt` — linear layer mapping BERT's embedding space into Llama's

Trained via `notebooks/02_classifier_explainer_pipeline_demo.ipynb` (and the
upstream training scripts in `reference/LogLLM-master/train.py`). If you
don't have these weights locally, either retrain them or host them on
Hugging Face Hub and point `CLASSIFIER_WEIGHTS_PATH` in `.env` at the
downloaded copy.

## `explainer/` — root-cause explanation model

Llama-3.2-3B-Instruct + a LoRA adapter, fine-tuned on `data/explainer_dataset/`
to generate a plain-English explanation for anomalies the classifier flags.
Trained in `notebooks/03_explainer_model_training.ipynb` (Unsloth + LoRA,
~25 minutes on an A100) and published to Hugging Face Hub — the backend
(`backend/app.py`, `backend/explainer/inference.py`) downloads it directly by
its `EXPLAINER_ADAPTER_REPO` repo id, so nothing needs to be stored locally
for inference. No local files are kept in this folder.
