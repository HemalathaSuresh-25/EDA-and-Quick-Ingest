#!/bin/bash
set -e

# ===============================
# CONFIG
# ===============================
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
TEMP_DIR="$PROJECT_DIR/temp_runtime"
MODEL_RUNTIME_DIR="$TEMP_DIR/models"
MODEL_SOURCE_DIR="$PROJECT_DIR/models_1"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
PREPARE_LOG="$TEMP_DIR/prepare.log"

MODE="$1"
INPUT_PATH="$2"

mkdir -p "$TEMP_DIR"
mkdir -p "$MODEL_RUNTIME_DIR"

# ===============================
# USAGE
# ===============================
usage() {
    echo "Usage:"
    echo "  $0 install"
    echo "  $0 prepare"
    echo "  $0 analyze <log_file>"
    exit 1
}

[ -z "$MODE" ] && usage

# ===============================
# INSTALL MODE
# ===============================
if [ "$MODE" = "install" ]; then
    echo "======================================"
    echo " ATTEST-EDA | INSTALL MODE"
    echo "======================================"

    if ! command -v python3 &>/dev/null; then
        echo "ERROR: Python3 not found"
        exit 1
    fi

    echo "✔ Python: $(python3 --version)"

    if [ ! -d "$VENV_DIR" ]; then
        echo "▶ Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip

    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "ERROR: requirements.txt not found"
        exit 1
    fi

    echo "▶ Installing dependencies..."
    pip install -r "$REQUIREMENTS_FILE"

    chmod +x "$0"

    echo "======================================"
    echo " INSTALL COMPLETED SUCCESSFULLY"
    echo "======================================"
    exit 0
fi

# ===============================
# ACTIVATE VENV
# ===============================
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found. Run install first."
    exit 1
fi

# ===============================
# COPY MODELS
# ===============================
copy_models_if_missing() {
    REQUIRED_FILES=(
        rootcause_fail_classifier.pkl
        rootcause_abort_classifier.pkl
        tfidf_rootcause_fail.pkl
        tfidf_rootcause_abort.pkl
    )

    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MODEL_RUNTIME_DIR/$f" ]; then
            cp "$MODEL_SOURCE_DIR/$f" "$MODEL_RUNTIME_DIR/"
        fi
    done
}

# ===============================
# PREPARE MODE
# ===============================
if [ "$MODE" = "prepare" ]; then
    START_TIME=$(date +%s)
    START_HUMAN=$(date)

    {
        echo "======================================"
        echo " ATTEST-EDA | PREPARE MODE"
        echo " Start Time : $START_HUMAN"
        echo "======================================"

        python "$SCRIPTS_DIR/preprocess_logs.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/feature_engineering.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/failure_clustering.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/abort_clustering.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/merge_clusters.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/root_cause_tagging.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/create_train_test_split.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/train_status_model.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/train_rootcause_fail_model.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/train_rootcause_abort_model.py" --out "$TEMP_DIR"
        python "$SCRIPTS_DIR/freeze_status_preprocessing.py" --out "$TEMP_DIR"

        END_TIME=$(date +%s)
        END_HUMAN=$(date)
        DURATION=$((END_TIME - START_TIME))

        echo "--------------------------------------"
        echo " End Time   : $END_HUMAN"
        echo " Duration   : ${DURATION} seconds"
        echo " PREPARE STATUS : SUCCESS"
        echo "======================================"
        echo
    } | tee -a "$PREPARE_LOG"

    touch "$TEMP_DIR/.prepared"
    exit 0
fi

# ===============================
# ANALYZE MODE
# ===============================
if [ "$MODE" = "analyze" ]; then
    [ -z "$INPUT_PATH" ] && usage

    if [ ! -f "$INPUT_PATH" ]; then
        echo "ERROR: Log file not found → $INPUT_PATH"
        exit 1
    fi

    copy_models_if_missing

    echo "======================================"
    echo " ATTEST-EDA | ANALYZE MODE"
    echo "======================================"
    echo "Log File : $INPUT_PATH"
    echo "--------------------------------------"

    python "$SCRIPTS_DIR/log_analyzer.py" \
        --log "$INPUT_PATH" \
        --runtime "$TEMP_DIR"

    echo "======================================"
    echo " ANALYSIS COMPLETED"
    echo "======================================"
    exit 0
fi

usage
