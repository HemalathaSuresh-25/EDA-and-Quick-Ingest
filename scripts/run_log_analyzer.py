#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, sys, os
from joblib import load
from scipy.sparse import hstack
import numpy as np

CONFIDENCE_THRESHOLD = 0.60
PASS_OVERRIDE_RATIO  = 0.60

status_model     = load("models/status_classifier_xgb.joblib")
vectorizer_word  = load("models/tfidf_vectorizer_word.pkl")
vectorizer_char  = load("models/tfidf_vectorizer_char.pkl")
label_encoder    = load("models/label_encoder.joblib")

# CLEAN TEXT FOR ML

def clean_for_model(text):
    text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}", " ", text)
    text = re.sub(r"[^\w\s\-:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# EXPLICIT PASS/FAIL/ABORT DETECTION

def extract_explicit_status(text):
    txt = text.lower()
    abort = len(re.findall(r"\babort(?:ed|s)?\b", txt))
    fail  = len(re.findall(r"\bfail(?:ed|ure|s)?\b", txt))
    pas   = len(re.findall(r"\bpass(?:ed|es)?\b", txt))

    if abort > 0:
        return "ABORT"
    if fail > 0 and pas == 0:
        return "FAIL"
    if pas > 0 and fail == 0:
        return "PASS"

    return None

# FAILURE PATTERNS

failure_patterns = {
    r"timeout|timed out|no response" :
        ("Timeout / No Response", "Increase timeout & verify DUT responsiveness"),

    r"disconnect|link down|connection lost|peer lost" :
        ("Link Failure", "Check cable, port state & network reachability"),

    r"segfault|core dump|crash|panic|fatal" :
        ("Software Crash", "Review crash logs / memory utilization"),

    r"config fail|invalid config|invalid parameter|bad parameter|mismatch" :
        ("Configuration Error", "Correct configuration fields & re-run"),

    r"packet drop|rx miss|tx miss|lost packet|no packet" :
        ("Packet Drop", "Check MTU, traffic load & interface queues"),

    r"crc error|checksum failed|frame error" :
        ("Data Integrity Error", "Inspect signal quality / cables / SFP"),

    r"ptp sync fail|timestamp mismatch|offset too high|sync lost" :
        ("PTP Synchronization Loss", "Verify GM source, delay & clock domain"),

    r"sctp init|sctp chunk|init chunk fail" :
        ("SCTP INIT Packet Missing", "Check SCTP negotiation / chunk exchange"),

    r"capture.txt is empty|no packets captured" :
        ("No Packet Capture", "Verify traffic generator or capture interface"),
}

# ABORT PATTERNS

abort_patterns = {
    r"abort due to timeout|aborted due to timeout" :
        ("Timeout Abort", "Increase timeout or check DUT responsiveness"),

    r"user abort|manual abort|operator abort|aborted by user" :
        ("User/Operator Abort", "Ensure operator intervention is required"),

    r"environment abort|setup abort|initialization abort" :
        ("Environment Abort", "Check DUT setup, environment variables & connections"),

    r"script aborted|execution aborted|automation aborted" :
        ("Script Abort", "Check automation script flow, steps & dependencies"),

    r"resource abort|memory abort|cpu abort" :
        ("Resource Abort", "Check system CPU/memory limits & usage"),

    r"dependency abort|prerequisite abort" :
        ("Dependency Abort", "Verify config files & dependencies"),
}

# GENERIC EXTRACTIONS

generic_reason_map = {
    "link": "Check network connectivity & ports",
    "timeout": "Increase timeout or check DUT responsiveness",
    "crash": "Investigate logs, memory & core dumps",
    "segfault": "Review software stability & environment",
    "config": "Validate configuration parameters",
    "packet": "Check network traffic, MTU & interfaces",
    "ptp": "Verify PTP sync, GM source & delay",
    "sctp": "Check SCTP negotiation & chunk exchange",
    "capture": "Verify capture interface or traffic generator",
    "announce": "Check DUT announcement message transmission",
}

# UNIFIED REASON DETECTION
def detect_reason(log, status):

    text = log.lower()

    # FAIL REASON EXTRACTION

    if status == "FAIL":

        # "# Result: FAILED ..."
        m = re.search(r"#\s*result\s*:\s*failed\s*(.+)", log, re.IGNORECASE)
        if m:
            reason = m.group(1).strip()
            reason = re.sub(r"\s*#+\s*$", "", reason)

            reco = "Verify DUT activity & test conditions"
            for key, rec in generic_reason_map.items():
                if key in reason.lower():
                    reco = rec
                    break

            return reason, reco

        # Pattern-based
        for pattern, (reason, reco) in failure_patterns.items():
            if re.search(pattern, text):
                return reason, reco

        # Last "test case failed"
        m2 = re.findall(r"test case failed.*", text)
        if m2:
            reason = m2[-1].strip()
            reco = "Verify DUT activity & test conditions"

            for key, rec in generic_reason_map.items():
                if key in reason.lower():
                    reco = rec
                    break

            return reason, reco

    # ABORT REASON EXTRACTION

    if status == "ABORT":

        # Pattern-based
        for pattern, (reason, reco) in abort_patterns.items():
            if re.search(pattern, text):
                return reason, reco

        # Last abort line → CLEAN IT
        m3 = re.findall(r".*abort.*", text)
        if m3:
            raw_reason = m3[-1].strip()

            # Remove timestamps
            raw_reason = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}", "", raw_reason)

            # Remove leading "#", "-", spaces
            raw_reason = re.sub(r"^[#\-\s]+", "", raw_reason)

            # Remove "test case aborted:"
            raw_reason = raw_reason.replace("test case aborted:", "").strip()

            # Normalize spaces
            raw_reason = re.sub(r"\s+", " ", raw_reason).strip()

            return raw_reason, "Verify abort conditions & DUT environment"

    return None, None

# MAIN PROCESSOR

def analyze_log_file(logfile):

    raw = open(logfile, "r", errors="ignore").read()

    explicit = extract_explicit_status(raw)
    if explicit:
        final = explicit
    else:
        cleaned = clean_for_model(raw)
        X = hstack([
            vectorizer_word.transform([cleaned]),
            vectorizer_char.transform([cleaned])
        ])

        proba = status_model.predict_proba(X)[0]
        classes = label_encoder.classes_
        final = classes[np.argmax(proba)]
        conf = max(proba)

        if conf < CONFIDENCE_THRESHOLD:
            final = "UNCERTAIN"

        # Override FAIL → PASS if high pass tokens
        if final == "FAIL":
            pc = len(re.findall(r"\bpass", raw, re.I))
            fc = len(re.findall(r"\bfail", raw, re.I))
            if pc + fc > 0 and pc / (pc + fc) >= PASS_OVERRIDE_RATIO:
                final = "PASS"

    # OUTPUT

    print("\n========== LOG RESULT ==========")
    print(f"FILE        : {os.path.basename(logfile)}")
    print(f"PREDICTED   : {final}")

    if final in ["FAIL", "ABORT"]:
        reason, reco = detect_reason(raw, final)

        print("\n--- DETAILS ---")
        if reason:
            print(f"REASON         : {reason}")
            print(f"RECOMMENDATION : {reco}")
        else:
            print("REASON         : Not recognized")
            print("RECOMMENDATION : new pattern for this log")

    print("================================\n")

# ENTRY POINT
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_log_analyzer.py <logfile>")
        sys.exit(0)

    analyze_log_file(sys.argv[1])
