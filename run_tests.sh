#!/usr/bin/env bash
# Mimics the GitHub Actions PythonTests workflow locally.
# Usage:
#   ./run_tests.sh           # run tests, regenerate Julia reference data only if missing
#   ./run_tests.sh --regen   # force-regenerate Julia reference data before testing
set -euo pipefail

REGEN=0
for arg in "$@"; do
    [[ "$arg" == "--regen" ]] && REGEN=1
done

cd "$(dirname "$0")"

# Step 1: Julia reference data
if [[ $REGEN -eq 1 ]] || [[ ! -f tests/julia_reference_data.json ]] || [[ ! -f julia_solar_synthesis.h5 ]]; then
    echo "==> Generating Julia reference data..."
    julia --project=. tests/generate_julia_reference.jl
    julia --project=. tests/generate_solar_synthesis_reference.jl
else
    echo "==> Julia reference data already present (use --regen to refresh)"
fi

# Step 2: CI artifact placeholders
echo "==> Setting up CI artifact placeholders..."
python .github/scripts/setup_ci_artifacts.py

# Step 3: Tests
echo "==> Running tests..."
pytest tests/ -v --tb=short
