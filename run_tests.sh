#!/usr/bin/env bash
# Mimics the GitHub Actions PythonTests workflow locally.
# Usage:
#   ./run_tests.sh           # run tests, regenerate Julia reference data only if missing
#   ./run_tests.sh --regen   # force-regenerate Julia reference data before testing
#   PYTEST_WORKERS=4 ./run_tests.sh   # override the worker count
#
# Runs on 8 workers by default. CI uses -n 1: the hosted runners have two cores,
# where the workers would contend rather than overlap. The JAX compilation cache
# in tests/conftest.py is what makes the parallel run worthwhile -- without it
# every worker recompiles the chemical-equilibrium kernel from scratch.
#
# --dist loadfile, not xdist's default --dist load. Much of this suite's cost is
# amortised in module- and class-scoped fixtures (plans, compiled gradients),
# and scattering one file's tests across workers rebuilds all of it per worker.
# It also runs the whole suite out of host memory: eight workers each holding
# their own copy of the reverse-mode synthesis program is std::bad_alloc.
set -euo pipefail

REGEN=0
for arg in "$@"; do
    [[ "$arg" == "--regen" ]] && REGEN=1
done

WORKERS="${PYTEST_WORKERS:-8}"

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
echo "==> Running tests on ${WORKERS} worker(s)..."
pytest tests/ -v --tb=short -n "${WORKERS}" --dist loadfile
