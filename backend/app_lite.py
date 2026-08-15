"""
LogSense — Lite API-Key Backend (no GPU required)
-----------------------------------------------------
A drop-in alternative to backend/app.py for demos, grading, or any environment
without a GPU to host the fine-tuned Llama models. Instead of running the
LogLLM classifier + explainer locally, this exposes the exact same
`/analyze` and `/live-log` contract the frontend expects, but delegates both
classification and explanation to a single hosted LLM call (Gemini by
default — this is the version actually used for live demos of this project).

Any OpenAI-compatible or Gemini-compatible provider works here: just set
LLM_PROVIDER and the matching API key in .env. To add another provider,
implement a new branch in `classify_and_explain()` below.

Run:
    uvicorn backend.app_lite:app --host 0.0.0.0 --port 8000
"""
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" or "openai_compatible"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY")  # e.g. an OpenRouter/Groq/OpenAI key
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HDFS_LOG_PATH = os.getenv("HDFS_LOG_PATH", os.path.join(REPO_ROOT, "data", "sample", "HDFS.log"))

app = FastAPI(title="LogSense API (lite / API-key mode)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(HDFS_LOG_PATH, "r") as f:
    HDFS_LOGS = [line.strip() for line in f if line.strip()]
_current_log_index = 0


class LogRequest(BaseModel):
    block_id: str
    semantic_text_sequence: list
    event_id_sequence: list
    latency: float


PROMPT_TEMPLATE = """You are an expert HDFS log anomaly detection system.

Analyze this HDFS log block.

Block ID:
{block_id}

Semantic Text Sequence:
{semantic_text_sequence}

Event ID Sequence:
{event_id_sequence}

Latency:
{latency} seconds

Classify the log block as either:
- Normal
- Anomalous

Then explain your reasoning in 3-5 sentences of technical prose.

IMPORTANT:
Return ONLY valid JSON. Do not use markdown. Do not wrap the JSON in ```.

Output format:
{{"status": "Normal", "explanation": "..."}}
OR
{{"status": "Anomalous", "explanation": "..."}}
"""


def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


def _call_openai_compatible(prompt: str) -> str:
    if not OPENAI_COMPAT_API_KEY:
        raise ValueError("OPENAI_COMPAT_API_KEY is not set. Add it to your .env file.")
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_COMPAT_API_KEY, base_url=OPENAI_COMPAT_BASE_URL)
    response = client.chat.completions.create(
        model=OPENAI_COMPAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def classify_and_explain(req: LogRequest) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        block_id=req.block_id,
        semantic_text_sequence=req.semantic_text_sequence,
        event_id_sequence=req.event_id_sequence,
        latency=req.latency,
    )

    if LLM_PROVIDER == "gemini":
        raw_text = _call_gemini(prompt)
    elif LLM_PROVIDER == "openai_compatible":
        raw_text = _call_openai_compatible(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' (use 'gemini' or 'openai_compatible').")

    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(cleaned)
        return {"status": result["status"], "explanation": result["explanation"]}
    except (json.JSONDecodeError, KeyError):
        return {"status": "Unknown", "explanation": cleaned}


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "LogSense API (lite)", "provider": LLM_PROVIDER}


@app.get("/live-log")
async def get_live_log():
    global _current_log_index
    log_line = HDFS_LOGS[_current_log_index]
    _current_log_index = (_current_log_index + 1) % len(HDFS_LOGS)
    return {"lvl": "INFO", "ts": datetime.now().strftime("%H:%M:%S"), "text": log_line}


@app.post("/analyze")
async def analyze_log(req: LogRequest):
    try:
        result = classify_and_explain(req)
        return {"block_id": req.block_id, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
