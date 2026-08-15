"""Centralized configuration for the LogSense backend, loaded from environment
variables / a local .env file (see .env.example at the repo root)."""
import os

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Secrets / tokens ---
HF_TOKEN = os.getenv("HF_TOKEN")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")  # optional: a reserved static ngrok domain

# --- Model identifiers ---
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")
CLASSIFIER_LLAMA_MODEL = os.getenv("CLASSIFIER_LLAMA_MODEL", "meta-llama/Meta-Llama-3-8B")
EXPLAINER_BASE_MODEL = os.getenv("EXPLAINER_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
EXPLAINER_ADAPTER_REPO = os.getenv("EXPLAINER_ADAPTER_REPO", "kovidritesh/Llama-3.2-3B-FineTuned")

# --- Local paths (override in .env if your data/models live elsewhere) ---
CLASSIFIER_WEIGHTS_PATH = os.getenv(
    "CLASSIFIER_WEIGHTS_PATH", os.path.join(REPO_ROOT, "models", "classifier", "ft_model_HDFS")
)
HDFS_LOG_PATH = os.getenv("HDFS_LOG_PATH", os.path.join(REPO_ROOT, "data", "raw", "HDFS.log"))

# --- Server ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
