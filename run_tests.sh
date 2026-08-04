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

# Every reference fixture the test suite compares against, and the script that
# builds it. All of them come from the Korg.jl version pinned in Project.toml.
REFERENCES=(
    "tests/julia_reference_data.json:tests/generate_julia_reference.jl"
    "julia_solar_synthesis.h5:tests/generate_solar_synthesis_reference.jl"
    "julia_broad_solar_synthesis.h5:tests/generate_broad_solar_reference.jl"
    "tests/data/balmer_abo_reference.h5:tests/gen_balmer_abo_reference.jl"
)

# Step 1: Julia environment. The exact-version pin in Project.toml means this
# resolves to the targeted Korg.jl release or fails loudly.
echo "==> Instantiating Julia environment..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.status("Korg")'

# Step 2: Julia reference data
for entry in "${REFERENCES[@]}"; do
    target="${entry%%:*}"
    script="${entry#*:}"
    if [[ $REGEN -eq 1 ]] || [[ ! -f "$target" ]]; then
        echo "==> Generating $target..."
        julia --project=. "$script"
    else
        echo "==> $target already present (use --regen to refresh)"
    fi
done

# Step 3: CI artifact placeholders
echo "==> Setting up CI artifact placeholders..."
python .github/scripts/setup_ci_artifacts.py

# Step 4: Tests
echo "==> Running tests..."
pytest tests/ -v --tb=short
