#!/usr/bin/env python3
"""
HDFS Anomaly Explanation Generator — v3
Strictly follows the specification rules for generating natural language
explanations of anomalous HDFS log sequences.

Rules enforced:
  RULE 1 — No event ID codes (E5, E22, etc.) in explanation text
  RULE 2 — Don't start with "Type X anomaly" or "This sequence"
  RULE 3 — Cover: what signals the anomaly, root cause, cluster impact
  RULE 4 — 3–5 sentences, no bullet points, no markdown
  RULE 5 — Rotation for identical sequences
  RULE 6 — Incorporate TimeInterval/Latency when insightful
  RULE 7 — 0-indexed positions when referencing sequence locations
  RULE 8 — Skip invalid anomaly types, log warnings
"""

import csv
import json
import re
from collections import defaultdict, Counter

# ── Event phrase mapping (RULE 1: translate IDs to plain English) ──────────
EVENT_PHRASE = {
    "E1":  "adding an already existing block",
    "E2":  "block verification succeeded",
    "E3":  "served block to destination",
    "E4":  "exception while serving block to destination",
    "E5":  "DataNode block-receive initiation",
    "E6":  "full block receipt with size confirmation",
    "E7":  "writeBlock exception",
    "E8":  "PacketResponder hard interrupt",
    "E9":  "successful block receipt confirmation",
    "E10": "PacketResponder exception",
    "E11": "PacketResponder shutdown",
    "E12": "exception writing block to mirror",
    "E13": "empty packet received during transfer",
    "E14": "receiveBlock exception",
    "E15": "block file offset adjustment",
    "E16": "successful block transmission confirmation",
    "E17": "block transfer failure",
    "E18": "replication thread launch to secondary node",
    "E19": "block file reopened",
    "E20": "block metadata not found in DataNode volume map",
    "E21": "block file deletion",
    "E22": "NameNode block allocation",
    "E23": "block added to invalidation set",
    "E24": "block removed from neededReplications",
    "E25": "NameNode instruction to replicate block to another DataNode",
    "E26": "replica registration in the NameNode block map",
    "E27": "redundant block registration",
    "E28": "block registered with no parent file",
    "E29": "replication timeout by PendingReplicationMonitor",
}

# Events that signal actual failure
FAILURE_EVENTS = {"E7", "E8", "E10", "E12", "E13", "E14", "E17", "E20", "E27", "E28", "E29"}

# Context-dependent signal events — normal in some contexts, anomalous in others
CONTEXT_SIGNAL_EVENTS = {"E6", "E16", "E18", "E25", "E4"}

# Normal pipeline events (E11 and E9 together are normal, not failure)
NORMAL_PIPELINE = {"E9", "E11"}

# Valid anomaly types (RULE 8)
VALID_TYPES = {0, 1, 3, 4, 5, 7, 9, 16, 21, 31}

# ── Anomaly type definitions ──────────────────────────────────────────────
# Each type has: (what_signals, root_cause, cluster_impact)
TYPE_DEFS = {
    0: (
        "block was allocated and some replicas registered, but the pipeline was restarted "
        "mid-stream with extra replication steps — successful transmission and receipt events "
        "appear alongside re-replication instructions such as NameNode replication directives "
        "or replication thread launches, indicating the pipeline was renegotiated",
        "a DataNode temporarily dropped from the pipeline causing the NameNode to reissue "
        "replication instructions, and although recovery was attempted the final state "
        "remained anomalous",
        "partial replicas may exist on intermediate nodes, increasing the risk of stale "
        "replica serving and read inconsistencies across the cluster",
    ),
    1: (
        "a redundant block registration request arrived at the NameNode after the block "
        "was already registered in the block map — the duplicate addStoredBlock signals "
        "a race condition in the registration protocol",
        "a stale acknowledgment or delayed RPC from a desynchronized DataNode triggered "
        "a duplicate registration attempt at the NameNode",
        "minor metadata bloat on the NameNode, though the effective replication factor "
        "remains unaffected and no data is at risk",
    ),
    3: (
        "the NameNode allocated the block and a DataNode opened a receive pipeline, but "
        "no replicas were ever registered in the block map — the sequence contains "
        "allocation and pipeline setup events with no replica registration following",
        "the client or upstream DataNode terminated the write pipeline before any data "
        "was committed, possibly from a client timeout, application crash, or lease expiration",
        "a wasted block lease and potentially orphaned temporary files on the DataNode, "
        "though no persistent metadata corruption occurs on the NameNode",
    ),
    4: (
        "after a normal-looking replication pipeline, an addStoredBlock request arrived "
        "for a block that no longer belongs to any file in the namespace — the block "
        "was stored but orphaned",
        "block data was written to disk but the file metadata was never finalized on "
        "the NameNode, leaving an orphaned replica without a parent file reference",
        "orphaned data consuming storage space across DataNodes, requiring manual "
        "scavenge or periodic cleanup to reclaim capacity",
    ),
    5: (
        "replicas were registered then immediately invalidated and deleted, but the "
        "deletion failed because the BlockInfo metadata entry was missing from the "
        "DataNode volume map — the volume map lookup failure during cleanup signals "
        "a metadata inconsistency",
        "the NameNode accepted replicas then invalidated them, likely from a lease "
        "expiration or race condition, while the DataNode had already lost or never "
        "persisted the metadata entry for this block",
        "wasted write I/O as replicas are created and immediately discarded; the "
        "DataNode may hold orphaned data that the NameNode believes was cleaned up",
    ),
    7: (
        "the DataNode encountered an exception in the receiveBlock path while receiving "
        "block data through the write pipeline — this exception signals that the data "
        "stream was interrupted or corrupted mid-transfer",
        "a network transmission error or transient disk I/O failure on the receiving "
        "DataNode corrupted or truncated the data stream during the block transfer",
        "truncated replicas will fail checksum validation, requiring complete "
        "re-replication from a healthy source node and increasing cluster recovery load",
    ),
    9: (
        "multiple distinct exception types occurred in a cascading failure pattern — "
        "writeBlock exceptions, PacketResponder exceptions, and receiveBlock exceptions "
        "appear together, indicating a severe pipeline disruption across multiple components",
        "a primary failure on a receiving DataNode cascaded through the entire pipeline, "
        "triggering exception paths across different components — possibly a DataNode JVM "
        "crash or disk failure mid-write",
        "multiple partial replicas and inconsistent metadata across DataNodes that all "
        "require reconciliation through full re-replication, with elevated data loss risk",
    ),
    16: (
        "a writeBlock exception occurred early in the pipeline and no replica was ever "
        "registered on the NameNode — the sequence contains failure events but no "
        "successful receipt confirmation or replica registration follows",
        "a dropped connection or downstream DataNode crash mid-transfer broke the "
        "replication chain before any data could be acknowledged by the recipient",
        "no replicas exist for this block allocation, requiring a full re-write from "
        "the client application with no recovery possible from existing cluster state",
    ),
    21: (
        "during block deletion the DataNode could not find the BlockInfo metadata entry "
        "in its volume map — the volume map lookup failure signals that the block's "
        "metadata was never properly persisted or was already removed",
        "the block was never registered in the local metadata store on the DataNode, or "
        "the metadata was removed before the physical deletion request arrived from "
        "the NameNode",
        "orphaned block files may remain on disk while the NameNode believes cleanup "
        "succeeded, silently wasting storage space and requiring manual reconciliation",
    ),
    31: (
        "a writeBlock exception occurred immediately after pipeline setup with no "
        "replicas ever registered — the sequence contains only allocation and pipeline "
        "initiation events before the exception terminates the write",
        "a hard pipeline failure on the receiving DataNode from a disk error, "
        "out-of-memory condition, or JVM crash during the initial write operation",
        "no replicas exist for this allocation, requiring a full re-write from the "
        "client application with no intermediate data to salvage",
    ),
}

# ── Utility functions ─────────────────────────────────────────────────────

def load_templates(path):
    """Load HDFS_log_templates.csv and verify event reference."""
    templates = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            eid = row[0].strip()
            if eid.startswith("E"):
                templates[eid] = row[1].strip()
    return templates


def load_sequences(path):
    """Load Event_traces.csv, filter to Fail rows with valid types."""
    seqs = []
    skipped_warnings = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        bi = hdr.index("BlockId")
        li = hdr.index("Label")
        ti = hdr.index("Type")
        fi = hdr.index("Features")
        ci = hdr.index("TimeInterval")
        lat_i = hdr.index("Latency")

        for row in rdr:
            if len(row) <= max(bi, li, ti, fi):
                continue

            # Filter to Fail rows
            if row[li].strip() != "Fail":
                continue

            # Parse features
            fs = row[fi].strip()
            if not fs or fs == "[]":
                skipped_warnings.append((row[bi].strip(), "empty features"))
                continue

            events = [
                e.strip().strip("\"' ")
                for e in fs.strip("[]").split(",")
                if e.strip()
            ]
            if not events:
                skipped_warnings.append((row[bi].strip(), "empty after parsing"))
                continue

            # Parse anomaly type (RULE 8)
            try:
                atype = int(row[ti].strip())
            except (ValueError, IndexError):
                skipped_warnings.append((row[bi].strip(), f"unparseable type: {row[ti].strip()}"))
                continue

            if atype not in VALID_TYPES:
                skipped_warnings.append((row[bi].strip(), f"invalid type: {atype}"))
                continue

            # Parse time interval
            ti_str = (row[ci] or "").strip()
            time_interval = []
            if ti_str and ti_str != "[]":
                try:
                    time_interval = json.loads(ti_str)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parse latency
            latency = None
            lat_str = (row[lat_i] or "").strip()
            if lat_str:
                try:
                    latency = float(lat_str)
                except ValueError:
                    pass

            seqs.append({
                "block_id": row[bi].strip(),
                "anomaly_type": atype,
                "sequence": events,
                "time_interval": time_interval,
                "latency": latency,
            })

    return seqs, skipped_warnings


# ── Analysis helpers ──────────────────────────────────────────────────────

def analyze_sequence(seq, time_interval, latency):
    """Analyze a sequence and return structured analysis results."""
    result = {
        "length": len(seq),
        "latency": latency,
        "time_interval": time_interval,
    }

    # Collect positions of each event type
    event_positions = {}
    for eid in EVENT_PHRASE:
        hits = [i for i, e in enumerate(seq) if e == eid]
        if hits:
            event_positions[eid] = hits
    result["event_positions"] = event_positions

    # Count normal E11→E9 cycles
    norm_cycles = 0
    i = 0
    while i < len(seq) - 1:
        if seq[i] == "E11" and seq[i + 1] == "E9":
            norm_cycles += 1
            i += 2
        else:
            i += 1
    result["norm_cycles"] = norm_cycles

    # Failure events with positions
    failures = [(pos, eid) for pos, eid in enumerate(seq) if eid in FAILURE_EVENTS]
    result["failures"] = failures
    result["fail_count"] = len(failures)

    # E26 → immediate E23 (immediate invalidation after registration)
    result["imm_inval"] = any(
        i < len(seq) - 1 and seq[i] == "E26" and seq[i + 1] == "E23"
        for i in range(len(seq) - 1)
    )

    # Allocated but no registration
    result["alloc_no_reg"] = "E22" in event_positions and "E26" not in event_positions

    # Count serving operations
    result["serve_count"] = len(event_positions.get("E3", []))

    # E20 count
    result["e20_count"] = len(event_positions.get("E20", []))

    # First failure position
    if failures:
        result["first_fail_pos"] = min(pos for pos, _ in failures)
        result["last_fail_pos"] = max(pos for pos, _ in failures)
        result["pre_fail_normal"] = sum(
            1 for i in range(result["first_fail_pos"])
            if seq[i] in {"E5", "E9", "E11"}
        )
    else:
        result["first_fail_pos"] = None
        result["pre_fail_normal"] = 0

    # Time insights (RULE 6)
    time_insight = None
    time_detail = None

    if latency is not None:
        if latency < 5:
            time_insight = "very_short"
            time_detail = (
                f"the entire trace completed in only {latency}s, "
                f"indicating the failure occurred almost immediately after pipeline setup"
            )
        elif latency > 3600 and len(seq) >= 4:
            time_insight = "very_long"
            time_detail = (
                f"the total latency of {int(latency)}s (over an hour) "
                f"suggests the block sat in a pending or limbo state before resolution"
            )

    if time_insight is None and time_interval and len(time_interval) > 1:
        # Check for long gap before deletion events
        del_indices = [
            i for i, e in enumerate(seq)
            if e in ("E21", "E23") and i < len(time_interval)
        ]
        if del_indices:
            max_del_gap = max(time_interval[i] for i in del_indices if i < len(time_interval))
            if max_del_gap > 60:
                time_insight = "long_gap_before_cleanup"
                time_detail = (
                    "a prolonged gap before the cleanup events suggests the block "
                    "sat in an inconsistent state before deletion was attempted"
                )

    if time_insight is None and time_interval and len(time_interval) > 1:
        max_gap = max(time_interval)
        if max_gap > 120:
            time_insight = "long_gap"
            time_detail = (
                "inter-event gaps of over two minutes between some steps "
                "suggest network delays or processing stalls"
            )
        elif max_gap > 30:
            time_insight = "moderate_gap"
            time_detail = (
                "moderate inter-event gaps indicate possible queuing delays "
                "on the DataNode"
            )

    result["time_insight"] = time_insight
    result["time_detail"] = time_detail

    return result


def count_sentences(text):
    """Count sentences in a text."""
    sents = [s.strip() for s in re.split(r"\.(?:\s+|$)", text) if s.strip()]
    return len(sents)


# ── Explanation generation ────────────────────────────────────────────────

# Rotation counter: keyed by (type, tuple(sequence))
rotation_counter = defaultdict(int)


def build_failure_description(analysis, seq):
    """Build a human-readable description of what signals the anomaly (RULE 1)."""
    parts = []
    fails = analysis["failures"]
    fc = analysis["fail_count"]

    if fails:
        for pos, eid in fails[:3]:
            parts.append(f"{EVENT_PHRASE[eid]} at position {pos}")
        if fc > 3:
            extra = fc - 3
            parts.append(f"plus {extra} additional failure signal{'s' if extra > 1 else ''}")
        return "; ".join(parts)

    # Check for context-dependent signal events
    context_signals = [(i, e) for i, e in enumerate(seq) if e in CONTEXT_SIGNAL_EVENTS]
    if context_signals:
        for pos, eid in context_signals[:3]:
            parts.append(f"{EVENT_PHRASE[eid]} at position {pos}")
        if len(context_signals) > 3:
            extra = len(context_signals) - 3
            parts.append(f"plus {extra} additional context-dependent signal{'s' if extra > 1 else ''}")
        return "; ".join(parts)

    if analysis["alloc_no_reg"]:
        return "block allocation with no subsequent replica registration"
    elif analysis["imm_inval"]:
        return "registration immediately invalidated without intervening block serving"
    else:
        return "an event ordering that deviates from the canonical HDFS replication lifecycle"


def build_normal_context(analysis):
    """Build a phrase about normal activity that occurred in the sequence."""
    nn = ""
    if analysis["norm_cycles"] > 0:
        nn = f"{analysis['norm_cycles']} normal pipeline cycles"
    sc = analysis["serve_count"]
    if sc >= 5:
        if nn:
            nn += f" and {sc} serving operations"
        else:
            nn = f"{sc} serving operations"
    return nn


def build_extra_detail(analysis, seq):
    """Build an extra detail snippet based on analysis."""
    parts = []
    e20c = analysis["e20_count"]
    ep = analysis["event_positions"]

    if e20c >= 2:
        parts.append(
            f"the volume map lookup failed {e20c} times during deletion, "
            f"indicating a systemic metadata problem across multiple blocks"
        )
    elif analysis["imm_inval"] and "E26" in ep:
        reg_pos = min(ep["E26"])
        parts.append(
            f"replica registration at position {reg_pos} was immediately "
            f"rolled back by invalidation at position {reg_pos + 1}"
        )
    elif analysis["alloc_no_reg"]:
        alloc_pos = min(ep["E22"])
        parts.append(
            f"block allocation at position {alloc_pos} was never completed "
            f"with a replica registration"
        )
    elif analysis["fail_count"] >= 2 and analysis["first_fail_pos"] is not None:
        fp0 = analysis["first_fail_pos"]
        fpl = analysis["last_fail_pos"]
        if fpl > fp0 + 1:
            parts.append(
                f"failure signals span positions {fp0} through {fpl} "
                f"in a {analysis['length']}-event trace"
            )
    elif analysis["norm_cycles"] == 0 and analysis["pre_fail_normal"] == 0 and analysis["length"] <= 3:
        parts.append(
            f"the trace is only {analysis['length']} events long, with "
            f"the anomaly occurring near the start of the block lifecycle"
        )

    return parts


def generate_explanation(atype, seq, time_interval, latency):
    """Generate a natural-language explanation for one anomalous sequence."""
    # Rotation key for identical sequences (RULE 5)
    key = (atype, tuple(seq))
    rotation_counter[key] += 1
    rot = (rotation_counter[key] - 1) % 5

    analysis = analyze_sequence(seq, time_interval, latency)
    what_signals, root_cause, impact = TYPE_DEFS[atype]

    # Build components
    fail_desc = build_failure_description(analysis, seq)
    normal_ctx = build_normal_context(analysis)
    extra_parts = build_extra_detail(analysis, seq)
    time_detail = analysis["time_detail"]

    # Build the explanation based on rotation angle (RULE 5)
    sentences = []

    if rot == 0:
        # Focus on what events signal the anomaly
        if normal_ctx:
            sentences.append(
                f"Despite {normal_ctx} occurring within the sequence, "
                f"the anomaly is signaled by {fail_desc}."
            )
        else:
            sentences.append(
                f"The failure is signaled by {fail_desc}."
            )
        sentences.append(f"The root cause is that {root_cause}.")
        sentences.append(f"At the cluster level, {impact}.")

    elif rot == 1:
        # Focus on root cause
        sentences.append(f"The root cause traces to {root_cause}.")
        if normal_ctx:
            sentences.append(
                f"Even though {normal_ctx} were observed, the evidence "
                f"points to {fail_desc}."
            )
        else:
            sentences.append(
                f"The evidence comes from {fail_desc}."
            )
        sentences.append(f"Cluster-wide, {impact}.")

    elif rot == 2:
        # Focus on cluster impact
        sentences.append(f"From a cluster perspective, {impact}.")
        sentences.append(f"The anomaly is triggered by {fail_desc}.")
        sentences.append(f"Underlying this is {root_cause}.")

    elif rot == 3:
        # Combine root cause and recovery implications
        ip = impact[0].lower() + impact[1:] if impact else impact
        if ip.endswith("."):
            ip = ip[:-1]
        sentences.append(f"The root cause is {root_cause}.")
        sentences.append(f"Recovery implications: {ip}.")
        sentences.append(f"The failure is evident from {fail_desc}.")

    else:  # rot == 4
        # Focus on operational/monitoring perspective
        sentences.append(
            f"From a monitoring standpoint, this anomaly is flagged by {fail_desc}."
        )
        sentences.append(f"The underlying root cause is {root_cause}.")
        sentences.append(f"The impact on cluster health is that {impact}.")

    # Add extra detail if available
    if extra_parts:
        sentences.append(extra_parts[0].capitalize() + ".")

    # Add time insight if available (RULE 6)
    if time_detail:
        # Try to integrate naturally
        time_sent = time_detail[0].upper() + time_detail[1:] + "."
        sentences.append(time_sent)

    # Build the full explanation
    explanation = " ".join(sentences)

    # Clean up double spaces
    explanation = re.sub(r" {2,}", " ", explanation)

    # Ensure sentence count is 3-5 (RULE 4)
    sent_count = count_sentences(explanation)
    if sent_count < 3:
        # Pad with additional insight
        if analysis["failures"]:
            first = analysis["failures"][0]
            extra = (
                f"The first anomaly indicator is "
                f"{EVENT_PHRASE[first[1]]} at position {first[0]}."
            )
        elif analysis["length"] <= 5:
            extra = (
                f"The sequence spans only {analysis['length']} events, "
                f"indicating the anomaly occurred very early in the block lifecycle."
            )
        else:
            extra = (
                f"The trace contains {analysis['length']} events "
                f"with {analysis['fail_count']} failure indicators."
            )
        explanation += " " + extra

    # Trim to max 5 sentences if needed
    sents = [s.strip() for s in re.split(r"\.(?:\s+|$)", explanation) if s.strip()]
    if len(sents) > 5:
        explanation = ". ".join(sents[:5]) + "."

    # Final cleanup
    explanation = re.sub(r" {2,}", " ", explanation)

    return explanation


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import os

    print("=" * 60)
    print("HDFS Anomaly Explanation Generator — v3")
    print("=" * 60)

    # Load templates (for verification)
    templates = load_templates("HDFS.log_templates.csv")
    print(f"\nLoaded {len(templates)} event templates from HDFS.log_templates.csv")

    # Load and filter sequences
    print("\nLoading anomalous sequences from Event_traces.csv...")
    sequences, skipped = load_sequences("Event_traces.csv")
    print(f"  Found {len(sequences)} valid anomalous sequences")

    # Log skipped rows
    if skipped:
        skip_types = Counter(reason for _, reason in skipped)
        print(f"  Skipped {len(skipped)} rows:")
        for reason, count in sorted(skip_types.items()):
            print(f"    - {reason}: {count}")

    # Generate explanations
    print("\nGenerating explanations...")
    output = []
    for i, s in enumerate(sequences):
        if i > 0 and i % 2000 == 0:
            print(f"  {i}/{len(sequences)}...")
        explanation = generate_explanation(
            s["anomaly_type"],
            s["sequence"],
            s.get("time_interval"),
            s.get("latency"),
        )
        output.append({
            "block_id": s["block_id"],
            "anomaly_type": s["anomaly_type"],
            "sequence": s["sequence"],
            "explanation": explanation,
        })

    # Write output
    output_path = "hdfs_explanations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  Written: {len(output)} entries ({size_mb:.2f} MB) to {output_path}")

    # ── Validation ──────────────────────────────────────────────────────

    # 1. Type breakdown
    type_counts = Counter(e["anomaly_type"] for e in output)
    print(f"\n  Type breakdown:")
    for t in sorted(type_counts):
        print(f"    Type {t}: {type_counts[t]}")

    # 2. Unique sequences and explanations
    unique_sequences = len(set(tuple(e["sequence"]) for e in output))
    unique_explanations = len(set(e["explanation"] for e in output))
    print(f"\n  Unique sequences: {unique_sequences}")
    print(f"  Unique explanations: {unique_explanations} "
          f"({unique_explanations / max(len(output), 1) * 100:.1f}%)")

    # 3. Sentence count distribution
    sent_counts = [count_sentences(e["explanation"]) for e in output]
    sent_dist = Counter(sent_counts)
    print(f"  Sentence distribution: {dict(sorted(sent_dist.items()))}")
    print(f"  Min={min(sent_counts)} Max={max(sent_counts)} "
          f"Avg={sum(sent_counts) / len(sent_counts):.1f}")

    outside_range = sum(1 for c in sent_counts if c < 3 or c > 5)
    print(f"  Outside 3-5 sentence range: {outside_range} "
          + ("[PASS]" if outside_range == 0 else "[WARN]"))

    # 4. RULE 1 check: no raw event IDs in explanations
    raw_ids_found = set()
    for e in output:
        ids_in_expl = re.findall(r'\bE\d+\b', e["explanation"])
        raw_ids_found.update(ids_in_expl)
    print(f"  Raw event IDs in explanations: "
          + ("[PASS] none" if not raw_ids_found else f"[FAIL] {sorted(raw_ids_found)}"))

    # 5. RULE 2 check: don't start with "Type X"
    starts_with_type = sum(1 for e in output if e["explanation"].startswith("Type "))
    print(f"  Start with 'Type ': {starts_with_type}/{len(output)} "
          + ("[PASS]" if starts_with_type == 0 else "[FAIL]"))

    # 6. RULE 4 check: no markdown or bullet points
    has_bullets = sum(1 for e in output if "- " in e["explanation"] or "* " in e["explanation"])
    print(f"  With bullet points: {has_bullets}/{len(output)} "
          + ("[PASS]" if has_bullets == 0 else "[FAIL]"))

    # 7. Sample explanations per type
    print("\n  Sample explanations:")
    for t in sorted(VALID_TYPES):
        samples = [e for e in output if e["anomaly_type"] == t]
        if samples:
            print(f"\n    Type {t}:")
            print(f"      {samples[0]['explanation'][:250]}")

    # 8. Rotation validation: show 3 explanations for same type+sequence
    seen = {}
    for e in output:
        key = (e["anomaly_type"], tuple(e["sequence"]))
        if key not in seen:
            seen[key] = []
        seen[key].append(e)
        if len(seen[key]) >= 3:
            print(f"\n  Rotation sample (type {key[0]}):")
            for j, ex in enumerate(seen[key][:3]):
                print(f"    [{j}] {ex['explanation'][:180]}...")
            break

    # 9. Skipped rows report
    print(f"\n  Rows skipped: {len(skipped)}")
    for block_id, reason in skipped[:10]:
        print(f"    - {block_id}: {reason}")
    if len(skipped) > 10:
        print(f"    ... and {len(skipped) - 10} more")

    print("\nDone.")


if __name__ == "__main__":
    main()
