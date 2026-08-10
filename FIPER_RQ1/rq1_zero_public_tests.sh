#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]] && SCRIPT_DIR="."
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd -P)"
cd "$REPO_ROOT"

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  --help|-h)
    cat <<'EOF'
Usage: bash rq1_zero_public_tests.sh [--dry-run]

Configuration is supplied through environment variables:
  INPUT_CSV        English-schema RQ1 input CSV
  OUTPUT_DIR       Pipeline output directory
  PROVIDER         openai, gemini, deepseek, or codellama
  MODEL            Provider model identifier or local checkpoint
  LANGUAGE         python or cpp
  PYTHON_BIN       Python executable (default: python3)
  GENERATION_THREADS, EVALUATION_THREADS, NUM_CANDIDATES
  WARMUP_RUNS, MEASURED_RUNS, TEST_TIMEOUT_SECONDS
EOF
    exit 0
    ;;
  "") ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROVIDER="${PROVIDER:-deepseek}"
MODEL="${MODEL:-deepseek-v3.2-exp}"
LANGUAGE="${LANGUAGE:-python}"
INPUT_CSV="${INPUT_CSV:-$REPO_ROOT/data/rq1/${LANGUAGE}_input.csv}"
MODEL_TAG="${MODEL//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/rq1/${LANGUAGE}_${MODEL_TAG}}"
GENERATION_THREADS="${GENERATION_THREADS:-4}"
EVALUATION_THREADS="${EVALUATION_THREADS:-4}"
NUM_CANDIDATES="${NUM_CANDIDATES:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-25}"
TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-2}"

if [[ "$LANGUAGE" != "python" && "$LANGUAGE" != "cpp" ]]; then
  echo "LANGUAGE must be python or cpp." >&2
  exit 2
fi

required_files=(
  generate_candidates.py
  evaluate_candidates.py
  test_synthesis/process_generated_tests.py
  candidate_selection/select_best_candidate.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 2; }
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
  [[ -f "$INPUT_CSV" ]] || { echo "Input CSV not found: $INPUT_CSV" >&2; exit 2; }
  mkdir -p "$OUTPUT_DIR"
fi

run_step() {
  printf '>>'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

generated_inputs="$OUTPUT_DIR/01_generated_test_inputs.csv"
verified_tests="$OUTPUT_DIR/02_oracle_verified_tests.csv"
described_programs="$OUTPUT_DIR/03_semantic_descriptions.csv"

run_step "$PYTHON_BIN" generate_candidates.py \
  --stage tests --input "$INPUT_CSV" --output "$generated_inputs" \
  --provider "$PROVIDER" --model "$MODEL" --language "$LANGUAGE" \
  --threads "$GENERATION_THREADS" --num-candidates 1 --temperature 1.0

run_step "$PYTHON_BIN" -m test_synthesis.process_generated_tests \
  --input "$generated_inputs" --output "$verified_tests" --language "$LANGUAGE" \
  --max-tests 10 --timeout "$TEST_TIMEOUT_SECONDS" --threads "$EVALUATION_THREADS"

run_step "$PYTHON_BIN" generate_candidates.py \
  --stage description --input "$verified_tests" --output "$described_programs" \
  --provider "$PROVIDER" --model "$MODEL" --language "$LANGUAGE" \
  --threads "$GENERATION_THREADS" --num-candidates 1 --temperature 0.01

temperatures=(0.7 1.0 1.3 1.6)


round_input="$described_programs"
for round in 1 2 3 4; do
  generated="$OUTPUT_DIR/round_${round}_generated.csv"
  evaluated="$OUTPUT_DIR/round_${round}_evaluated.csv"
  selected="$OUTPUT_DIR/round_${round}_selected.csv"

  run_step "$PYTHON_BIN" generate_candidates.py \
    --stage optimize --input "$round_input" --output "$generated" \
    --provider "$PROVIDER" --model "$MODEL" --language "$LANGUAGE" \
    --threads "$GENERATION_THREADS" --num-candidates "$NUM_CANDIDATES" \
    --temperature "${temperatures[$((round - 1))]}"

  run_step "$PYTHON_BIN" evaluate_candidates.py \
    --input "$generated" --output "$evaluated" --language "$LANGUAGE" \
    --warmup-runs "$WARMUP_RUNS" --measured-runs "$MEASURED_RUNS" \
    --timeout "$TEST_TIMEOUT_SECONDS" --threads "$EVALUATION_THREADS"

  run_step "$PYTHON_BIN" -m candidate_selection.select_best_candidate \
    --input "$evaluated" --output "$selected" --round "$round"

  round_input="$selected"
done

run_step cp "$round_input" "$OUTPUT_DIR/final_candidates.csv"
echo "RQ1 generation pipeline complete: $OUTPUT_DIR/final_candidates.csv"
