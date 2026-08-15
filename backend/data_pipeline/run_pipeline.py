"""
LogSense — End-to-End Pipeline (Stage 1 classifier -> Stage 2 explainer)
--------------------------------------------------------------------------
Parses a raw HDFS log, runs the fine-tuned LogLLM classifier (BERT + Llama-3-8B)
on every block, and generates natural-language root-cause explanations for
every block the classifier flags as anomalous using the Llama-3.2-3B explainer.

Designed to run on a GPU host (Kaggle/Colab/local CUDA machine — the base
models are large; see README for VRAM requirements). Configure paths via
environment variables (see .env.example) rather than editing this file.

Usage:
    python run_pipeline.py
"""
import ast
import gc
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set. Copy .env.example to .env and add your Hugging "
        "Face access token (needs read access + accepted Llama 3 / 3.2 licenses)."
    )
os.environ["HF_TOKEN"] = HF_TOKEN

# Make the classifier module importable when this script is run standalone.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "classifier"))
from custom_dataset import CustomCollator, CustomDataset  # noqa: E402
from model import LogLLM  # noqa: E402

# --- Configurable paths (env vars, with local project-relative defaults) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_HDFS_LOG = os.getenv("HDFS_LOG_PATH", os.path.join(REPO_ROOT, "data", "raw", "HDFS.log"))
TEMPLATES_CSV = os.getenv("HDFS_TEMPLATES_PATH", os.path.join(REPO_ROOT, "data", "sample", "HDFS_log_templates.csv"))
CLASSIFIER_WEIGHTS = os.getenv("CLASSIFIER_WEIGHTS_PATH", os.path.join(REPO_ROOT, "models", "classifier", "ft_model_HDFS"))

PARSED_CSV = "parsed_sessions.csv"
SYNTHETIC_TEST_CSV = "test.csv"
FINAL_EXPLANATIONS_CSV = "final_explanations.csv"

BASE_EXPLAINER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
EXPLAINER_ADAPTER_REPO = os.getenv("EXPLAINER_ADAPTER_REPO", "kovidritesh/Llama-3.2-3B-FineTuned")

# How many flagged anomalies to explain (None = all of them).
EXPLAIN_LIMIT = int(os.getenv("EXPLAIN_LIMIT", "10")) or None


def parse_hdfs_log(raw_log_path: str, templates_csv_path: str) -> pd.DataFrame:
    """Stage 0: parse the raw HDFS log into per-block event sequences."""
    print("1. Building semantic parser for LogLLM & the explainer model...")

    id_to_text = {}
    templates = []

    with open(templates_csv_path, "r", encoding="utf-8") as f:
        reader = pd.read_csv(f)
        for _, row in reader.iterrows():
            event_id = str(row["EventId"]).strip()
            template = str(row["EventTemplate"]).strip()

            clean_template = template.replace("[*]", "<*>")
            id_to_text[event_id] = clean_template

            parts = clean_template.split("<*>")
            escaped_parts = [re.escape(p) for p in parts]
            pattern_str = ".*?".join(escaped_parts).replace(r"\ ", r"\s+")

            templates.append((event_id, re.compile(pattern_str, re.IGNORECASE | re.DOTALL)))

    print(f"Loaded {len(templates)} templates with elastic regex.")

    sessions = defaultdict(list)
    block_pattern = re.compile(r"blk_-?\d+")
    time_pattern = re.compile(r"^(\d{6})\s+(\d{6})")

    matched_count = 0
    with open(raw_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            block_match = block_pattern.search(line)
            if not block_match:
                continue
            block_id = block_match.group(0)

            event_id = next((eid for eid, pat in templates if pat.search(line)), None)
            if not event_id:
                continue

            matched_count += 1
            semantic_text = id_to_text[event_id]
            ts_match = time_pattern.match(line)
            if ts_match:
                dt = datetime.strptime(ts_match.group(1) + ts_match.group(2), "%y%m%d%H%M%S")
                sessions[block_id].append((dt, event_id, semantic_text))
            else:
                sessions[block_id].append((None, event_id, semantic_text))

    print(f"Extracted {matched_count} events into {len(sessions)} unique block sequences.")
    print("2. Formatting sequences...")

    results = []
    for block_id, events in sessions.items():
        events_sorted = sorted(events, key=lambda x: x[0] if x[0] is not None else datetime.max)

        id_sequence = [e[1] for e in events_sorted]
        text_sequence = [e[2] for e in events_sorted]

        timestamps = [e[0] for e in events_sorted if e[0] is not None]
        latency = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) >= 2 else 0.0

        results.append({
            "block_id": block_id,
            "sequence": str(id_sequence),
            "text_sequence": str(text_sequence),
            "latency": latency,
        })

    return pd.DataFrame(results)


def format_for_logllm(seq: str) -> str:
    try:
        events = ast.literal_eval(seq)
        if isinstance(events, list):
            return " ;-; ".join(events)
    except (ValueError, SyntaxError):
        pass
    return str(seq).replace(",", " ;-; ")


def classify_anomalies(df_parsed: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: run the LogLLM (BERT + Llama-3-8B) classifier."""
    df_parsed["Content"] = df_parsed["text_sequence"].apply(format_for_logllm)
    df_parsed["Label"] = 0
    df_parsed[["Content", "Label"]].to_csv(SYNTHETIC_TEST_CSV, index=False)

    print("Loading dataset into LogLLM pipeline...")
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)
    collator = CustomCollator(bert_tokenizer, max_seq_len=128, max_content_len=100)
    dataset = CustomDataset(SYNTHETIC_TEST_CSV)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collator)

    print("Initializing LogLLM model...")
    model = LogLLM(
        Bert_path="bert-base-uncased",
        Llama_path="meta-llama/Meta-Llama-3-8B",
        ft_path=CLASSIFIER_WEIGHTS,
        is_train_mode=False,
    )
    model.eval()

    preds = []
    print("Running anomaly classification...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch["inputs"].items()}
            output_tokens = model(inputs, batch["seq_positions"])
            for out in output_tokens:
                text = model.Llama_tokenizer.decode(out, skip_special_tokens=True).lower()
                preds.append(1 if "anomalous" in text else 0)

    df_parsed["Prediction"] = preds
    df_anomalies = df_parsed[df_parsed["Prediction"] == 1].copy()
    print(f"Detection complete! LogLLM flagged {len(df_anomalies)} anomalous sequences.")

    print("Purging LogLLM from GPU memory...")
    del model, dataloader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return df_anomalies


def explain_anomalies(df_anomalies: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: run the Llama-3.2-3B explainer on flagged anomalies."""
    from unsloth import FastLanguageModel

    print("Loading explanation model...")
    expl_model, expl_tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_EXPLAINER_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map="auto",
        token=HF_TOKEN,
    )
    expl_model.load_adapter(EXPLAINER_ADAPTER_REPO, token=HF_TOKEN)
    expl_model = FastLanguageModel.for_inference(expl_model)

    rows = df_anomalies.head(EXPLAIN_LIMIT) if EXPLAIN_LIMIT else df_anomalies
    results = []
    print(f"Generating root-cause explanations for {len(rows)} blocks...")

    for _, row in rows.iterrows():
        prompt = (
            f"Below is an anomalous HDFS log sequence. Explain what went wrong.\n\n"
            f"### Sequence:\n{row['sequence']}\n\n### Latency:\n{row['latency']} sec\n\n### Explanation:\n"
        )
        inputs = expl_tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = expl_model.generate(**inputs, max_new_tokens=300, use_cache=True)
        full_output = expl_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        explanation = full_output.split("### Explanation:\n")[-1].strip()

        results.append({
            "block_id": row["block_id"],
            "sequence": row["sequence"],
            "latency": row["latency"],
            "explanation": explanation,
        })
        print(f"\n[Block ID]: {row['block_id']}\n[Explanation]: {explanation}\n{'-' * 50}")

    return pd.DataFrame(results)


def main():
    df_parsed = parse_hdfs_log(RAW_HDFS_LOG, TEMPLATES_CSV)
    if len(df_parsed) == 0:
        print("CRITICAL ERROR: 0 sequences matched. Check HDFS_LOG_PATH / HDFS_TEMPLATES_PATH.")
        return

    df_parsed.to_csv(PARSED_CSV, index=False)
    print(f"Success! Generated {SYNTHETIC_TEST_CSV} with {len(df_parsed)} sequences.")

    df_anomalies = classify_anomalies(df_parsed)
    if len(df_anomalies) == 0:
        print("No anomalies detected — nothing to explain.")
        return

    df_final = explain_anomalies(df_anomalies)
    df_final.to_csv(FINAL_EXPLANATIONS_CSV, index=False)
    print(f"Pipeline finished. Explanations saved to {FINAL_EXPLANATIONS_CSV}.")


if __name__ == "__main__":
    main()
