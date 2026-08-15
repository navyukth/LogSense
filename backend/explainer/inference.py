"""
LogSense — Explainer Model Inference
-------------------------------------
Loads Llama-3.2-3B-Instruct + the LoRA "explainer" adapter from Hugging Face
and generates a root-cause explanation for an HDFS block already flagged as
anomalous by the classifier (see backend/classifier/model.py).

Requires HF_TOKEN to be set in the environment (see .env.example) — Llama 3.2
is a gated model on Hugging Face, so the token must belong to an account that
has accepted Meta's license for it.
"""
import os

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set. Copy .env.example to .env and add your Hugging "
        "Face access token (needs read access + accepted Llama 3.2 license)."
    )

import torch
from huggingface_hub import login
from peft import PeftModel
from unsloth import FastLanguageModel

login(token=HF_TOKEN)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_REPO = os.getenv("EXPLAINER_ADAPTER_REPO", "kovidritesh/Llama-3.2-3B-FineTuned")
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 300

SYSTEM_PROMPT = (
    "You are an HDFS log analysis expert specializing in Apache Hadoop distributed "
    "storage systems. Given a block's event sequence, anomaly type, and latency, "
    "explain the anomaly in plain English. Your explanation must cover: "
    "(1) what events signal the anomaly and why their ordering is anomalous, "
    "(2) the likely root cause, and "
    "(3) the cluster-level impact. "
    "Write 3-5 sentences of natural technical prose. "
    "Do not use event ID codes like E5 or E22 in your explanation — "
    "translate them to plain English descriptions."
)


def load_model():
    print(f"Loading base model : {BASE_MODEL}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    print(f"Loading LoRA adapter: {ADAPTER_REPO}")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_REPO,
        token=HF_TOKEN,
    )

    device = next(model.parameters()).device
    if device.type != "cuda":
        print(f"Warning: model is on '{device}' — inference will be slow.")
    else:
        print(f"Model loaded on {device}")

    FastLanguageModel.for_inference(model)
    print("Model ready\n")

    return model, tokenizer


def build_user_message(block_id, anomaly_type, sequence, latency_str):
    seq_str = sequence if isinstance(sequence, str) else str(sequence)
    seq_len = (
        len(sequence)
        if not isinstance(sequence, str)
        else seq_str.count(",") + 1
    )

    return (
        f"Block: {block_id}\n"
        f"Anomaly Type: {anomaly_type}\n"
        f"Sequence: {seq_str}\n"
        f"Sequence Length: {seq_len}\n"
        f"Latency: {latency_str}"
    )


def explain(model, tokenizer, block_id, anomaly_type, sequence, latency_str):
    user_msg = build_user_message(block_id, anomaly_type, sequence, latency_str)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def explain_from_detection_output(model, tokenizer, detection_result):
    latency_sec = float(detection_result.get("latency", 0))

    if latency_sec < 60:
        latency_str = f"{latency_sec:.0f} seconds"
    elif latency_sec < 3600:
        latency_str = f"{latency_sec / 60:.1f} minutes"
    else:
        latency_str = f"{latency_sec / 3600:.1f} hours"

    return explain(
        model=model,
        tokenizer=tokenizer,
        block_id=detection_result["block_id"],
        anomaly_type=detection_result["anomaly_type"],
        sequence=detection_result["sequence"],
        latency_str=latency_str,
    )


if __name__ == "__main__":
    model, tokenizer = load_model()

    detection_output = {
        "block_id": "blk_-1608999687919862906",
        "anomaly_type": 5,
        "sequence": ["E5", "E22", "E5", "E11", "E9", "E11", "E9", "E26", "E26", "E23", "E21", "E20"],
        "latency": 3802,
    }

    print("=" * 70)
    print("INPUT")
    print("=" * 70)
    print(f"Block ID      : {detection_output['block_id']}")
    print(f"Anomaly Type  : {detection_output['anomaly_type']}")
    print(f"Sequence      : {detection_output['sequence']}")
    print(f"Latency       : {detection_output['latency']} sec")
    print("\nGenerating explanation...\n")

    explanation = explain_from_detection_output(model, tokenizer, detection_output)

    print("=" * 70)
    print("MODEL OUTPUT")
    print("=" * 70)
    print(explanation)
    print("=" * 70)
