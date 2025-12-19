import os
import sys
import re
import joblib
import numpy as np

MODEL_DIR = r"C:\Users\hemalatha\Desktop\attest-eda\models_1"

FAIL_MODEL  = os.path.join(MODEL_DIR, "rootcause_fail_classifier.pkl")
ABORT_MODEL = os.path.join(MODEL_DIR, "rootcause_abort_classifier.pkl")

TFIDF_FAIL  = os.path.join(MODEL_DIR, "tfidf_rootcause_fail.pkl")
TFIDF_ABORT = os.path.join(MODEL_DIR, "tfidf_rootcause_abort.pkl")

print(" Loading ML models...")

fail_model  = joblib.load(FAIL_MODEL)
abort_model = joblib.load(ABORT_MODEL)

tfidf_fail  = joblib.load(TFIDF_FAIL)
tfidf_abort = joblib.load(TFIDF_ABORT)

print("✔ Models loaded\n")


# =========================
# ROOT CAUSE → RECOMMENDATION
# =========================

FAIL_RECOMMENDATION_MAP = {
    "Configuration Mismatch":
        "Verify DUT configuration parameters and reapply the correct configuration.",

    "Missing Resource / Identifier":
        "Ensure all required resources (IDs, profiles, sessions) are created before execution.",

    "Packet Transmission Failure":
        "Check packet flow, network connectivity, and DUT transmit counters.",

    "Port / Interface Mismatch":
        "Validate the interface/port mapping between DUT and test configuration.",

    "Incorrect Test":
        "Review test logic and expected behavior for correctness.",

    "PTP Protocol Failure":
        "Validate PTP profile, domain, clock type, and message exchange.",

    "General DUT Communist Result / Assertion Failure":
        "Inspect DUT logs for assertion failures or unexpected responses.",

    "CLI Command Failurecation Failure":
        "Verify CLI command syntax, permissions, and DUT state.",

    "Bit / Field Validation Error":
        "Check protocol field values and bit-level correctness.",

    "Clock / Timing Type Mismatch":
        "Ensure clock type and timing profile match DUT capability."
}

ABORT_RECOMMENDATION_MAP = {
    "Media Stream Not Established":
        "Ensure required media streams are initialized before test execution.",

    "Profile / Standard Not Supported":
        "Verify DUT firmware supports the selected profile or standard.",

    "Transport / Messaging Failure":
        "Check communication channel and message delivery between test system and DUT.",

    "Test Not Applicable":
        "Confirm the test is applicable for the current DUT mode and configuration.",

    "Precondition / Setup Failure":
        "Verify all preconditions and setup steps are completed successfully.",

    "Invalid Test Configuration":
        "Review and correct invalid or unsupported test parameters.",

    "Transport Protocol Mismatch":
        "Ensure transport protocol (IPv4/IPv6, UDP/TCP) matches DUT settings.",

    "User / Registration Incomplete":
        "Ensure user/device registration completes before starting the test."
}


# =========================
# HELPERS
# =========================

def pad_features(X, expected):
    """Pad or trim TF-IDF features to match model expectation"""
    current = X.shape[1]
    if current == expected:
        return X
    elif current < expected:
        pad = expected - current
        return np.hstack([X.toarray(), np.zeros((1, pad))])
    else:
        return X[:, :expected]


def extract_status(log_text):
    tail = "\n".join(log_text.splitlines()[-50:])

    if re.search(r"Result:\s*PASSED", tail, re.I):
        return "PASS", "Testcase Passed"

    m = re.search(r"Aborted\s*:\s*(.+)", tail, re.I)
    if m:
        return "ABORT", m.group(1).strip()

    if re.search(r"\bfail(ed)?\b", tail, re.I):
        return "FAIL", "Testcase Failed"

    return "UNKNOWN", "Unknown Status"


def extract_error_text(log_text):
    lines = log_text.splitlines()
    errors = [
        l.strip() for l in lines
        if re.search(
            r"(error|fail|failed|timeout|abort|mismatch|exception|not supported|invalid|does not|unable)",
            l, re.I
        )
    ]
    return " ".join(errors[-30:]) if errors else "No error found"


def recommendation(status, root_cause):
    if status == "FAIL":
        return FAIL_RECOMMENDATION_MAP.get(
            root_cause, "Analyze DUT logs and test configuration."
        )
    if status == "ABORT":
        return ABORT_RECOMMENDATION_MAP.get(
            root_cause, "Verify test setup and execution conditions."
        )
    return "No action required"


# =========================
# MAIN
# =========================

def analyze_log(log_path):
    print(f" Analyzing log file:\n{log_path}\n")

    with open(log_path, "r", errors="ignore") as f:
        text = f.read()

    status, reason = extract_status(text)

    if status == "PASS":
        print(" Result")
        print("Status        : PASS")
        print("Reason        : Testcase Passed")
        print("Recommendation: No action required")
        return

    if status == "ABORT" and "Stopped By User" in reason:
        print(" Result")
        print("Status        : ABORT")
        print(f"Reason        : {reason}")
        print("Recommendation: No action required")
        return

    error_text = extract_error_text(text)

    if status == "FAIL":
        vec = tfidf_fail.transform([error_text])
        vec = pad_features(vec, fail_model.n_features_in_)
        root = fail_model.predict(vec)[0]

    elif status == "ABORT":
        vec = tfidf_abort.transform([error_text])
        vec = pad_features(vec, abort_model.n_features_in_)
        root = abort_model.predict(vec)[0]

    else:
        root = "Unknown"

    print(" Result")
    print(f"Status        : {status}")
    print(f"Reason        : {root}")
    print(f"Recommendation: {recommendation(status, root)}")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python log_analyzer.py <log_file.log>")
        sys.exit(1)

    analyze_log(sys.argv[1])
