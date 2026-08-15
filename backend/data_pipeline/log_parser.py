"""
LogSense — Stage 1: Log Parser
--------------------------------
Parses raw HDFS log lines into per-block event sequences
using HDFS_log_templates.csv.

Input : HDFS.log  (raw log file)
Output: parsed_sessions.csv  (block_id, sequence, latency)

Usage in Jupyter:
    Run this entire cell. Then set LOG_FILE and TEMPLATE_FILE paths.
"""

import re
import csv
import os
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — update these paths
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_FILE = "HDFS_log_templates.csv"   # path to your templates CSV
LOG_FILE      = "HDFS.log"                 # path to your raw HDFS log file
OUTPUT_FILE   = "parsed_sessions.csv"      # output path

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD TEMPLATES & BUILD REGEX TREE
# ─────────────────────────────────────────────────────────────────────────────

def load_templates(template_file):
    """
    Reads HDFS_log_templates.csv and converts each template into a regex.
    """
    templates = []

    with open(template_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_id = row["EventId"].strip()
            template = row["EventTemplate"].strip()

            # BULLETPROOF WILDCARD FIX
            # Split by the exact string <*>, escape the safe parts, and glue with .*?
            parts = template.split("<*>")
            escaped_parts = [re.escape(p) for p in parts]
            pattern_str = ".*?".join(escaped_parts)

            compiled = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
            templates.append((event_id, compiled))

    print(f"✅ Loaded {len(templates)} event templates")
    return templates


def match_event(line, templates):
    """
    Try each template regex against the log line.
    Returns the first matching event_id, or None if no match.
    """
    for event_id, pattern in templates:
        if pattern.search(line):
            return event_id
    return None

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXTRACT BLOCK ID FROM LOG LINE
# ─────────────────────────────────────────────────────────────────────────────

# Matches patterns like: blk_-1234567890 or blk_1234567890
BLOCK_ID_PATTERN = re.compile(r"blk_-?\d+")

def extract_block_id(line):
    """Extract the first block ID found in a log line."""
    match = BLOCK_ID_PATTERN.search(line)
    return match.group(0) if match else None

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PARSE TIMESTAMP FROM LOG LINE
# ─────────────────────────────────────────────────────────────────────────────

# HDFS log format: 081109 203518 143 INFO dfs.DataNode$DataXceiver: ...
# Date: YYMMDD  Time: HHMMSS
TIMESTAMP_PATTERN = re.compile(r"^(\d{6})\s+(\d{6})")

def extract_timestamp(line):
    """
    Parse timestamp from HDFS log line.
    Returns a datetime object or None.
    """
    match = TIMESTAMP_PATTERN.match(line)
    if not match:
        return None
    try:
        date_str = match.group(1)   # e.g. 081109
        time_str = match.group(2)   # e.g. 203518
        dt = datetime.strptime(date_str + time_str, "%y%m%d%H%M%S")
        return dt
    except ValueError:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PARSE THE FULL LOG FILE INTO SESSIONS
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(log_file, templates):
    """
    Reads the log file line by line.
    Groups events by block_id chronologically.
    Returns a dict: { block_id: [(timestamp, event_id), ...] }
    """
    sessions = defaultdict(list)   # block_id → list of (timestamp, event_id)
    unmatched = 0
    matched   = 0
    no_block  = 0
    total     = 0

    print(f"Parsing {log_file} ...")

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue

            block_id = extract_block_id(line)
            if not block_id:
                no_block += 1
                continue

            event_id = match_event(line, templates)
            if not event_id:
                unmatched += 1
                continue

            timestamp = extract_timestamp(line)
            sessions[block_id].append((timestamp, event_id))
            matched += 1

    print(f"✅ Parsing complete")
    print(f"   Total lines   : {total}")
    print(f"   Matched events: {matched}")
    print(f"   No block ID   : {no_block}")
    print(f"   Unmatched     : {unmatched}")
    print(f"   Unique blocks : {len(sessions)}")

    return sessions

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — COMPUTE LATENCY & BUILD SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

def build_sessions(sessions):
    """
    For each block, sort events by timestamp and compute latency
    (time delta in seconds between first and last event).
    Returns list of dicts ready for inference.
    """
    results = []

    for block_id, events in sessions.items():

        # Sort by timestamp (put None timestamps last)
        events_sorted = sorted(
            events,
            key=lambda x: x[0] if x[0] is not None else datetime.max
        )

        sequence  = [e[1] for e in events_sorted]
        timestamps = [e[0] for e in events_sorted if e[0] is not None]

        # Compute latency
        if len(timestamps) >= 2:
            latency = (timestamps[-1] - timestamps[0]).total_seconds()
        else:
            latency = 0.0

        results.append({
            "block_id" : block_id,
            "sequence" : sequence,
            "latency"  : latency,
        })

    return results

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_sessions(sessions, output_file):
    """Save parsed sessions to CSV."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["block_id", "sequence", "latency"])
        for s in sessions:
            writer.writerow([
                s["block_id"],
                str(s["sequence"]),
                s["latency"],
            ])
    print(f"✅ Saved {len(sessions)} sessions → {output_file}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_pipeline(log_file=LOG_FILE, template_file=TEMPLATE_FILE, output_file=OUTPUT_FILE):
    """
    Full Stage 1 pipeline:
        raw log → event sequences per block → CSV output
    """
    # 1. Load templates
    templates = load_templates(template_file)

    # 2. Parse log into raw sessions
    raw_sessions = parse_log(log_file, templates)

    # 3. Build sorted sequences with latency
    sessions = build_sessions(raw_sessions)

    # 4. Save to CSV
    save_sessions(sessions, output_file)

    return sessions   # also return for direct use in notebook

# ─────────────────────────────────────────────────────────────────────────────
# DEMO — parse and show first 3 blocks
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    sessions = parse_pipeline()

    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (first 3 blocks)")
    print("=" * 70)

    for s in sessions[:3]:
        print(f"\nBlock ID : {s['block_id']}")
        print(f"Sequence : {s['sequence']}")
        print(f"Latency  : {s['latency']} sec")
