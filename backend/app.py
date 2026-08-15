"""
LogSense — FastAPI Inference Server
--------------------------------------
Serves the two-stage LogSense pipeline over HTTP:
  1. POST /analyze   — classify a log sequence (LogLLM: BERT + Llama-3-8B),
                        and if it's flagged anomalous, generate a root-cause
                        explanation (LoRA-tuned Llama-3.2-3B).
  2. GET  /live-log   — stream a synthetic "live tail" of the HDFS sample log,
                        used by the frontend's live-monitor panel.
  3. GET  /            — health check.

This needs a CUDA GPU with enough VRAM for both models (see README for
requirements — it was developed/run on Kaggle's dual T4 GPUs). If you only
have one GPU, load both models with device_map="auto" and rely on 4-bit
quantization, or run the classifier and explainer as two separate processes.

Run locally:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000

Expose via ngrok (e.g. from a Kaggle/Colab notebook with no public IP):
    set NGROK_AUTHTOKEN in .env, then run this module directly:
    python -m backend.app
"""
import os
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizerFast, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), "classifier"))
from custom_dataset import CustomCollator, CustomDataset  # noqa: E402
from model import LogLLM  # noqa: E402

from backend import config

app = FastAPI(title="LogSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if config.HF_TOKEN:
    os.environ["HF_TOKEN"] = config.HF_TOKEN

# --- Load classifier (LogLLM: BERT + Llama-3-8B) ---------------------------
print("Loading classifier (LogLLM)...")
bert_tokenizer = BertTokenizerFast.from_pretrained(config.BERT_MODEL_NAME, do_lower_case=True)
collator = CustomCollator(bert_tokenizer, max_seq_len=128, max_content_len=100)

classifier_model = LogLLM(
    Bert_path=config.BERT_MODEL_NAME,
    Llama_path=config.CLASSIFIER_LLAMA_MODEL,
    ft_path=config.CLASSIFIER_WEIGHTS_PATH,
    is_train_mode=False,
    device="cuda:0",
)
classifier_model.eval()

# --- Load explainer (Llama-3.2-3B + LoRA adapter) ---------------------------
print("Loading explainer model...")
explainer_tokenizer = AutoTokenizer.from_pretrained(config.EXPLAINER_BASE_MODEL, token=config.HF_TOKEN)
explainer_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_explainer_model = AutoModelForCausalLM.from_pretrained(
    config.EXPLAINER_BASE_MODEL,
    quantization_config=explainer_bnb_config,
    device_map="auto",
    token=config.HF_TOKEN,
)
explainer_model = PeftModel.from_pretrained(
    base_explainer_model, config.EXPLAINER_ADAPTER_REPO, token=config.HF_TOKEN
)
explainer_model.eval()

print("Both models loaded.")

# --- Live-log tail state -----------------------------------------------------
with open(config.HDFS_LOG_PATH, "r") as f:
    HDFS_LOGS = [line.strip() for line in f if line.strip()]
_current_log_index = 0


class LogRequest(BaseModel):
    block_id: str
    semantic_text_sequence: list
    event_id_sequence: list
    latency: float


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "LogSense API"}


@app.get("/live-log")
async def get_live_log():
    global _current_log_index
    log_line = HDFS_LOGS[_current_log_index]
    _current_log_index = (_current_log_index + 1) % len(HDFS_LOGS)
    return {"lvl": "INFO", "ts": datetime.now().strftime("%H:%M:%S"), "text": log_line}


@app.post("/analyze")
async def analyze_log(req: LogRequest):
    temp_csv_path = "temp_inference_row.csv"
    try:
        formatted_content = " ;-; ".join(req.semantic_text_sequence)
        pd.DataFrame([{"Content": formatted_content, "Label": 0}]).to_csv(temp_csv_path, index=False)

        infer_dataset = CustomDataset(temp_csv_path)
        infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, collate_fn=collator)
        batch = next(iter(infer_dataloader))
        inputs = {k: v.to(classifier_model.device) for k, v in batch["inputs"].items()}
        seq_positions = batch["seq_positions"]

        is_anomaly = False
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_tokens = classifier_model(inputs, seq_positions)
            text = classifier_model.Llama_tokenizer.decode(output_tokens[0], skip_special_tokens=True).lower()
            is_anomaly = "anomalous" in text

        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

        if not is_anomaly:
            return {
                "block_id": req.block_id,
                "status": "Normal",
                "explanation": "System operating as expected. No critical anomalies detected in this block.",
            }

        prompt = (
            f"Below is an anomalous HDFS log sequence. Explain what went wrong.\n\n"
            f"### Sequence:\n{req.event_id_sequence}\n\n### Latency:\n{req.latency} sec\n\n### Explanation:\n"
        )
        exp_inputs = explainer_tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = explainer_model.generate(**exp_inputs, max_new_tokens=300, use_cache=True)
        full_output = explainer_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        explanation = full_output.split("### Explanation:\n")[-1].strip()

        return {"block_id": req.block_id, "status": "Anomalous", "explanation": explanation}

    except Exception as e:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        raise HTTPException(status_code=500, detail=str(e))


def _run_with_ngrok():
    """Entry point for hosts with no public IP (Kaggle/Colab): tunnels the
    local server through ngrok. Requires NGROK_AUTHTOKEN in .env."""
    import uvicorn
    from pyngrok import ngrok

    if not config.NGROK_AUTHTOKEN:
        raise ValueError("NGROK_AUTHTOKEN is not set — required to expose the server via ngrok.")

    ngrok.set_auth_token(config.NGROK_AUTHTOKEN)
    connect_kwargs = {"domain": config.NGROK_DOMAIN} if config.NGROK_DOMAIN else {}
    listener = ngrok.connect(addr=config.API_PORT, **connect_kwargs)
    print(f"Live API: {listener.public_url}")
    print("Paste this URL into frontend/.env as VITE_API_BASE_URL.")

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


if __name__ == "__main__":
    _run_with_ngrok()
