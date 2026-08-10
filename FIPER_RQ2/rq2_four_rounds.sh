#!/usr/bin/env bash

# Reproduce the four progressive RQ2 optimization rounds.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/rq2}"
DATASET_ID="${DATASET_ID:-PIE}"
LANGUAGE="${LANGUAGE:-python}"
MODEL="${MODEL:-DeepSeekChat}"
MODEL_ID="${MODEL_ID:-}"
BASELINE_PATH="${BASELINE_PATH:-}"
WORKERS="${WORKERS:-4}"
NUM_CANDIDATES="${NUM_CANDIDATES:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-25}"
TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${LANGUAGE}" == "python" ]]; then
    LANGUAGE_TAG="Py"
else
    LANGUAGE_TAG="Cpp"
fi

mkdir -p "${RESULT_DIR}"
if [[ -z "${BASELINE_PATH}" ]]; then
    BASELINE="${DATA_DIR}/${DATASET_ID}_${LANGUAGE_TAG}_040_${MODEL}__X.csv"
else
    BASELINE="${BASELINE_PATH}"
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${BASELINE}" ]]; then
    echo "Missing RQ2 input: ${BASELINE}" >&2
    echo "Set DATA_DIR, DATASET_ID, LANGUAGE, MODEL, or BASELINE_PATH to match the downloaded benchmark files." >&2
    exit 2
fi

LLM=("${PYTHON_BIN}" "${SCRIPT_DIR}/llm_generation.py")
EVALUATE=("${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_execution_time.py")
FILTER=("${PYTHON_BIN}" "${SCRIPT_DIR}/io_processing/filter_repair_remove_zero.py")
SELECT=("${PYTHON_BIN}" "${SCRIPT_DIR}/generated_code_selection/select_best_generated_io_code.py")
GENERATION_OPTIONS=(
    --workers "${WORKERS}"
    --language "${LANGUAGE}"
)
if [[ -n "${MODEL_ID}" ]]; then
    GENERATION_OPTIONS+=(--model "${MODEL_ID}")
fi

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

GENERATED_IO="${RESULT_DIR}/050_generated_io.csv"
VERIFIED_IO="${RESULT_DIR}/051_oracle_verified_io.csv"

run "${LLM[@]}" --baseline-df-path "${BASELINE}" --generated-df-path "${GENERATED_IO}" \
    --iteration-round -1 --num-candidates 1 --candidates-per-request 1 \
    --repetitions 1 --temperature 1.0 "${GENERATION_OPTIONS[@]}"
run "${FILTER[@]}" --dataset-path "${GENERATED_IO}" --output-path "${VERIFIED_IO}" \
    --language "${LANGUAGE}" --timeout-seconds "${TEST_TIMEOUT_SECONDS}" --workers "${WORKERS}"

OVERVIEW="${RESULT_DIR}/070_overview.csv"
run "${LLM[@]}" --baseline-df-path "${VERIFIED_IO}" --generated-df-path "${OVERVIEW}" \
    --iteration-round 0 --num-candidates 1 --candidates-per-request 1 \
    --repetitions 1 --temperature 0.01 "${GENERATION_OPTIONS[@]}"

CURRENT="${OVERVIEW}"
temperatures=(0.7 1.0 1.3 1.6)
for ROUND in 1 2 3 4; do
    GENERATED="${RESULT_DIR}/round_${ROUND}_generated.csv"
    EVALUATED="${RESULT_DIR}/round_${ROUND}_evaluated.csv"
    SELECTED="${RESULT_DIR}/round_${ROUND}_selected.csv"
    run "${LLM[@]}" --baseline-df-path "${CURRENT}" --generated-df-path "${GENERATED}" \
        --iteration-round "${ROUND}" --num-candidates "${NUM_CANDIDATES}" \
        --candidates-per-request 1 --repetitions "${NUM_CANDIDATES}" \
        --temperature "${temperatures[$((ROUND - 1))]}" "${GENERATION_OPTIONS[@]}"
    run "${EVALUATE[@]}" --iteration-round "${ROUND}" --dataset-path "${GENERATED}" \
        --output-path "${EVALUATED}" --language "${LANGUAGE}" \
        --warmup-runs "${WARMUP_RUNS}" --measured-runs "${MEASURED_RUNS}" \
        --timeout-seconds "${TEST_TIMEOUT_SECONDS}"
    run "${SELECT[@]}" --round "${ROUND}" --dataset-path "${EVALUATED}" --output-path "${SELECTED}"
    CURRENT="${SELECTED}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "RQ2 command preview completed; no files were generated."
else
    echo "RQ2 reproduction completed. Final results: ${CURRENT}"
fi
