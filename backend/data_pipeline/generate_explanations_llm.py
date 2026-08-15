"""
LogSense — Synthetic Explanation Dataset Generator (LLM-based)
-----------------------------------------------------------------
Generates the training data for the explainer model: for every anomalous
HDFS block in Event_traces.csv, asks an LLM (via an OpenAI-compatible API,
default OpenRouter) to write a natural-language root-cause explanation, then
writes {system, user, assistant} chat-format records to a JSONL file ready
for fine-tuning.

Event_traces.csv columns: BlockId, Label, Type, Features, TimeInterval, Latency
  - BlockId   : block identifier
  - Type      : NaN = normal block, numeric = anomaly type (filter key)
  - Features  : event sequence as string "[E5,E22,E5,...]"
  - Latency   : total latency in seconds (context for generation)

HDFS_log_templates.csv columns: EventId, EventTemplate

Configure via environment variables (see .env.example):
  OPENROUTER_API_KEY   required
  LLM_MODEL             default: meta-llama/llama-3.3-70b-instruct:free
  LLM_BASE_URL           default: https://openrouter.ai/api/v1

Usage:
    pip install openai pandas tqdm python-dotenv
    python generate_explanations_llm.py
"""

import ast
import hashlib
import json
import os
import time
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

EVENT_TRACES_PATH = os.getenv(
    "EVENT_TRACES_PATH", os.path.join(REPO_ROOT, "data", "raw", "preprocessed", "Event_traces.csv")
)
TEMPLATES_PATH = os.getenv(
    "HDFS_TEMPLATES_PATH", os.path.join(REPO_ROOT, "data", "sample", "HDFS_log_templates.csv")
)
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "logsense_generated.jsonl")
REJECTED_PATH = os.getenv("REJECTED_PATH", "logsense_rejected.jsonl")

MAX_SAMPLES = 700          # total anomalous blocks to process
MIN_SEQ_LENGTH = 3         # skip trivially short sequences
MAX_SEQ_LENGTH = 60        # skip absurdly long ones (e.g. E3 spam blocks)
MIN_EXPLANATION_LEN = 150  # characters — reject lazy/short outputs
DELAY_BETWEEN_CALLS = 3.0  # seconds — stay under API rate limits
TEMPERATURE = 0.85         # high enough for diversity, grounded enough to be correct
# ─────────────────────────────────────────────────────────────────────────

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def parse_sequence(raw: str) -> list[str]:
    """Parse '[E5,E22,E5,...]' string into ['E5','E22','E5',...]"""
    try:
        parsed = ast.literal_eval(raw.strip())
        return [str(e).strip() for e in parsed if str(e).strip()]
    except (ValueError, SyntaxError):
        raw = raw.strip().strip("[]")
        return [e.strip() for e in raw.split(",") if e.strip()]


def format_latency(latency) -> str:
    try:
        secs = float(latency)
        if secs < 60:
            return f"{secs:.0f} seconds"
        elif secs < 3600:
            return f"{secs / 60:.1f} minutes"
        else:
            return f"{secs / 3600:.1f} hours"
    except (TypeError, ValueError):
        return "unknown"


def build_prompt(block_id: str, sequence: list[str], templates: dict[str, str],
                  anomaly_type: str, latency_str: str) -> str:
    unique_events = list(dict.fromkeys(sequence))  # preserve order, deduplicate
    relevant_templates = {eid: templates[eid] for eid in unique_events if eid in templates}
    template_block = "\n".join(f"  {eid}: {tmpl}" for eid, tmpl in relevant_templates.items())
    seq_str = " -> ".join(sequence)

    rep = Counter(sequence)
    repeated = {e: c for e, c in rep.items() if c > 1}
    repetition_note = ""
    if repeated:
        repetition_note = "\nRepeated events: " + ", ".join(
            f"{e} appears {c}x" for e, c in sorted(repeated.items(), key=lambda x: -x[1])
        )

    return f"""You are a senior HDFS (Hadoop Distributed File System) systems engineer.
Explain what went wrong with this specific HDFS block based on its event lifecycle.

EVENT MEANINGS (what each event type means):
{template_block}

BLOCK DETAILS:
  Block ID     : {block_id}
  Anomaly Type : {anomaly_type}
  Total Latency: {latency_str}
  Sequence     : {seq_str}{repetition_note}

INSTRUCTIONS:
- Write 3 to 5 sentences of technical prose.
- Trace the causal chain: what triggered the failure, how it progressed, what the final state is.
- Reference specific events by their plain English meaning (not their E-codes like E5 or E22).
- If events repeat, explain what the repetition implies about the failure mode.
- If latency is unusually high or low, mention what that implies.
- Be specific to THIS block's sequence. Do not write a generic explanation.
- Do NOT start with "Here is", "Certainly", or any preamble. Write the explanation directly.
"""


def call_llm(prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=350,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = (attempt + 1) * 4
            print(f"  [retry {attempt + 1}] error: {e} — waiting {wait}s")
            time.sleep(wait)
    return None


def is_quality(explanation: str) -> tuple[bool, str]:
    if len(explanation) < MIN_EXPLANATION_LEN:
        return False, f"too short ({len(explanation)} chars)"

    bad_starts = ["here is", "certainly", "sure,", "as an ai", "i cannot"]
    lower = explanation.lower()
    for phrase in bad_starts:
        if lower.startswith(phrase):
            return False, f"bad opener: '{phrase}'"

    sentences = [s.strip() for s in explanation.split(".") if len(s.strip()) > 20]
    if len(sentences) < 2:
        return False, "fewer than 2 real sentences"

    return True, "ok"


def load_already_processed(path: str) -> set[str]:
    processed: set[str] = set()
    if not os.path.exists(path):
        return processed
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "block_id" in rec:
                    processed.add(rec["block_id"])
                    continue
                user_content = rec["messages"][1]["content"]
                for part in user_content.split("\n"):
                    if part.startswith("Block:"):
                        processed.add(part.split(":", 1)[1].strip())
                        break
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    return processed


def main():
    print("Loading data...")

    tmpl_df = pd.read_csv(TEMPLATES_PATH)
    templates: dict[str, str] = dict(
        zip(tmpl_df["EventId"].astype(str).str.strip(), tmpl_df["EventTemplate"].astype(str).str.strip())
    )
    print(f"  {len(templates)} event templates loaded")

    df = pd.read_csv(EVENT_TRACES_PATH)
    print(f"  {len(df)} total blocks")

    anomalies = df[df["Type"].notna()].copy()
    print(f"  {len(anomalies)} anomalous blocks (Type is not NaN)")

    anomalies["_seq"] = anomalies["Features"].apply(parse_sequence)
    anomalies["_seqlen"] = anomalies["_seq"].apply(len)

    filtered = anomalies[
        (anomalies["_seqlen"] >= MIN_SEQ_LENGTH) & (anomalies["_seqlen"] <= MAX_SEQ_LENGTH)
    ].copy()
    print(f"  {len(filtered)} after sequence length filter ({MIN_SEQ_LENGTH}-{MAX_SEQ_LENGTH})")

    filtered["_seq_hash"] = filtered["_seq"].apply(lambda s: hashlib.md5(",".join(s).encode()).hexdigest())
    filtered = filtered.drop_duplicates(subset=["_seq_hash"])
    print(f"  {len(filtered)} after sequence deduplication")

    sample = filtered.sample(n=min(MAX_SAMPLES, len(filtered)), random_state=42).reset_index(drop=True)
    print(f"\nGenerating explanations for {len(sample)} blocks...\n")

    type_dist = Counter(str(int(t)) for t in sample["Type"])
    print("Anomaly type distribution in sample:")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  Type {t}: {c}")
    print()

    already_processed = load_already_processed(OUTPUT_PATH) | load_already_processed(REJECTED_PATH)
    if already_processed:
        print(f"  Resuming — skipping {len(already_processed)} already processed blocks")

    sample = sample[~sample["BlockId"].astype(str).isin(already_processed)]
    print(f"  {len(sample)} blocks remaining to process\n")

    accepted_count = 0
    rejected_count = 0

    out_file = open(OUTPUT_PATH, "a")
    rej_file = open(REJECTED_PATH, "a")

    try:
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Generating"):
            block_id = str(row["BlockId"])
            sequence = row["_seq"]
            anomaly_type = str(int(row["Type"]))
            latency_str = format_latency(row.get("Latency", 0))

            prompt = build_prompt(block_id, sequence, templates, anomaly_type, latency_str)
            explanation = call_llm(prompt)

            if explanation is None:
                rej_file.write(json.dumps({
                    "block_id": block_id, "sequence": sequence, "reason": "LLM call failed after retries",
                }) + "\n")
                rej_file.flush()
                rejected_count += 1
                continue

            ok, reason = is_quality(explanation)

            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an HDFS log analysis expert. Given a block's event sequence, "
                            "anomaly type, and latency, explain the anomaly in plain English. "
                            "Cover: (1) what events signal the anomaly, (2) the likely root cause, "
                            "(3) the cluster-level impact. Write 3-5 sentences of technical prose."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Block: {block_id}\nAnomaly Type: {anomaly_type}\nSequence: {sequence}\n"
                            f"Sequence Length: {len(sequence)}\nLatency: {latency_str}"
                        ),
                    },
                    {"role": "assistant", "content": explanation},
                ]
            }

            if ok:
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                os.fsync(out_file.fileno())
                accepted_count += 1
            else:
                record["_reject_reason"] = reason
                rej_file.write(json.dumps(record) + "\n")
                rej_file.flush()
                rejected_count += 1

            time.sleep(DELAY_BETWEEN_CALLS)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved. Re-run to continue from here.")

    finally:
        out_file.close()
        rej_file.close()

    total_in_file = sum(1 for line in open(OUTPUT_PATH) if line.strip())

    print(f"\n{'=' * 50}")
    print(f"  This session accepted : {accepted_count}")
    print(f"  This session rejected : {rejected_count}")
    print(f"  Total in {OUTPUT_PATH:<20}: {total_in_file}")
    print(f"{'=' * 50}")

    if total_in_file < 200:
        print("\nUnder 200 total accepted. Check the rejected-output file.")
        print("Try lowering MIN_EXPLANATION_LEN or MAX_SEQ_LENGTH.")


if __name__ == "__main__":
    main()
