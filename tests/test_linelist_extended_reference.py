"""
Extended pytest tests for linelist functions against Julia reference data.

Covers:
- Line class construction (all fields) via Julia reference
- approximate_radiative_gamma: per-case reference comparison and physics
- approximate_gammas: per-case reference comparison and physics
- Line with explicit broadening parameters
- JIT compatibility
- Physics sanity checks (non-reference)

Reference data lives in tests/julia_reference_data.json.
"""

import json
import math
from pathlib import Path

import korg  # noqa: F401 — enables JAX x64 mode before anything else

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REFERENCE_FILE = Path(__file__).parent / "julia_reference_data.json"


@pytest.fixture(scope="module")
def reference_data():
    """Load Julia reference data once per module."""
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"Julia reference data not found at {REFERENCE_FILE}. "
            "Run: julia --project=/tmp/Korg.jl tests/generate_julia_reference.jl"
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_approx_gammas_key(key: str):
    """
    Parse a key like '5.0e-5_Fe I_1.01' into (wl, species_str, E_lower).
    The species name may contain a space (e.g. 'Fe I', 'Ca II').
    """
    # Split rightmost underscore to get E_lower
    right_split = key.rsplit("_", 1)
    E_lower = float(right_split[1])
    remainder = right_split[0]
    # Split leftmost underscore to get wl
    left_split = remainder.split("_", 1)
    wl = float(left_split[0])
    species_str = left_split[1]  # e.g. "Fe I" or "Ca II"
    return wl, species_str, E_lower


def _parse_arg_key(key: str):
    """Parse a key like '5.0e-5_-1.5' into (wl, log_gf)."""
    parts = key.split("_")
    return float(parts[0]), float(parts[1])


# ===========================================================================
# TestLineClassExtendedJuliaReference
# ===========================================================================

class TestLineClassExtendedJuliaReference:
    """Compare create_line results against Julia reference for all stored cases."""

    def _get_ref(self, reference_data):
        if "line_class" not in reference_data:
            pytest.skip("line_class reference data not available")
        return reference_data["line_class"]

    def test_wavelength_all_cases(self, reference_data):
        """Wavelength (in cm) must match Julia for every stored test case."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            julia = outputs[key]
            line = create_line(wl, log_gf, species_str, E_lower)
            assert np.isclose(line.wl, julia["wl"], rtol=1e-10), (
                f"wl mismatch case {key} ({species_str}): "
                f"Python={line.wl}, Julia={julia['wl']}"
            )

    def test_log_gf_all_cases(self, reference_data):
        """log_gf must be stored without modification."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            julia = outputs[key]
            line = create_line(wl, log_gf, species_str, E_lower)
            assert np.isclose(line.log_gf, julia["log_gf"], rtol=1e-10), (
                f"log_gf mismatch case {key}: Python={line.log_gf}, Julia={julia['log_gf']}"
            )

    def test_E_lower_all_cases(self, reference_data):
        """E_lower must be stored exactly."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            julia = outputs[key]
            line = create_line(wl, log_gf, species_str, E_lower)
            assert np.isclose(line.E_lower, julia["E_lower"], rtol=1e-10), (
                f"E_lower mismatch case {key}: Python={line.E_lower}, Julia={julia['E_lower']}"
            )

    def test_species_charge_all_cases(self, reference_data):
        """Species charge must match Julia."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            julia = outputs[key]
            line = create_line(wl, log_gf, species_str, E_lower)
            assert line.species.charge == julia["species_charge"], (
                f"species_charge mismatch case {key} ({species_str}): "
                f"Python={line.species.charge}, Julia={julia['species_charge']}"
            )

    def test_gamma_rad_all_cases(self, reference_data):
        """Approximate radiative gamma must match Julia within 1e-5 relative."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            julia = outputs[key]
            line = create_line(wl, log_gf, species_str, E_lower)
            assert np.isclose(line.gamma_rad, julia["gamma_rad"], rtol=1e-5), (
                f"gamma_rad mismatch case {key} ({species_str}): "
                f"Python={line.gamma_rad}, Julia={julia['gamma_rad']}"
            )

    def test_gamma_stark_all_cases(self, reference_data):
        """Approximate Stark gamma must match Julia within 1e-5 relative."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            julia = outputs[key]
            line = create_line(wl, log_gf, species_str, E_lower)
            assert np.isclose(line.gamma_stark, julia["gamma_stark"], rtol=1e-5), (
                f"gamma_stark mismatch case {key} ({species_str}): "
                f"Python={line.gamma_stark}, Julia={julia['gamma_stark']}"
            )

    def test_vdW_all_cases(self, reference_data):
        """van der Waals tuple must match Julia within 1e-5 relative."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            julia = outputs[key]
            line = create_line(wl, log_gf, species_str, E_lower)
            julia_vdW = julia["vdW"]
            assert np.isclose(line.vdW[0], julia_vdW[0], rtol=1e-5), (
                f"vdW[0] mismatch case {key} ({species_str}): "
                f"Python={line.vdW[0]}, Julia={julia_vdW[0]}"
            )
            assert np.isclose(line.vdW[1], julia_vdW[1], rtol=1e-10), (
                f"vdW[1] mismatch case {key} ({species_str}): "
                f"Python={line.vdW[1]}, Julia={julia_vdW[1]}"
            )

    def test_gamma_rad_physically_reasonable(self, reference_data):
        """Approximate gamma_rad should be > 1e6 for typical stellar lines."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            line = create_line(wl, log_gf, species_str, E_lower)
            assert line.gamma_rad > 1e6, (
                f"gamma_rad={line.gamma_rad} is unexpectedly small for {species_str}"
            )

    def test_log_gamma_vdW_physically_reasonable(self, reference_data):
        """log10(gamma_vdW) should be < -5 (gamma < 1e-5 per H atom) for all cases."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            line = create_line(wl, log_gf, species_str, E_lower)
            # vdW[1] == -1 means simple gamma; vdW[0] is the actual gamma value
            if line.vdW[1] == -1.0 and line.vdW[0] > 0:
                log_vdW = math.log10(line.vdW[0])
                assert log_vdW < -5, (
                    f"log_gamma_vdW={log_vdW:.3f} seems too large for {species_str}"
                )

    def test_line_object_vs_create_line(self, reference_data):
        """Constructing a Line directly with the same params as create_line gives the same results."""
        from korg.linelist import Line, create_line
        from korg.species import Species

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl_A, log_gf, species_str, E_lower) in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue

            # Use create_line to compute all broadening params
            line_created = create_line(wl_A, log_gf, species_str, E_lower)

            # Now construct explicitly with Line(...) using the same data
            wl_cm = wl_A * 1e-8
            species = Species(species_str)
            line_explicit = Line(
                wl=line_created.wl,
                log_gf=line_created.log_gf,
                species=species,
                E_lower=line_created.E_lower,
                gamma_rad=line_created.gamma_rad,
                gamma_stark=line_created.gamma_stark,
                vdW=line_created.vdW,
            )

            assert line_explicit.wl == line_created.wl
            assert line_explicit.log_gf == line_created.log_gf
            assert line_explicit.E_lower == line_created.E_lower
            assert line_explicit.gamma_rad == line_created.gamma_rad
            assert line_explicit.gamma_stark == line_created.gamma_stark
            assert line_explicit.vdW == line_created.vdW
            assert line_explicit.species.charge == line_created.species.charge


# ===========================================================================
# TestApproximateRadiativeGammaExtended
# ===========================================================================

class TestApproximateRadiativeGammaExtended:
    """Per-case reference tests for approximate_radiative_gamma."""

    def test_all_reference_cases(self, reference_data):
        """Every (wl, log_gf) pair in the JSON must match Julia to rtol=1e-6."""
        from korg.linelist import approximate_radiative_gamma

        if "approximate_radiative_gamma" not in reference_data:
            pytest.skip("approximate_radiative_gamma reference data not available")

        outputs = reference_data["approximate_radiative_gamma"]["outputs"]
        for key, julia_val in outputs.items():
            wl, log_gf = _parse_arg_key(key)
            py_val = float(approximate_radiative_gamma(wl, log_gf))
            assert np.isclose(py_val, julia_val, rtol=1e-6), (
                f"Mismatch wl={wl}, log_gf={log_gf}: Python={py_val}, Julia={julia_val}"
            )

    def test_positive_log_gf_gives_larger_gamma(self, reference_data):
        """gamma_rad should grow with log_gf at fixed wavelength."""
        from korg.linelist import approximate_radiative_gamma

        wl = 5e-5
        gamma_weak = float(approximate_radiative_gamma(wl, -3.0))
        gamma_strong = float(approximate_radiative_gamma(wl, 1.0))
        assert gamma_strong > gamma_weak, (
            "Higher log_gf should give larger gamma_rad"
        )

    def test_shorter_wavelength_gives_larger_gamma(self, reference_data):
        """gamma_rad ∝ λ^{-2}, so shorter λ → larger gamma."""
        from korg.linelist import approximate_radiative_gamma

        log_gf = 0.0
        gamma_blue = float(approximate_radiative_gamma(3e-5, log_gf))
        gamma_red = float(approximate_radiative_gamma(7e-5, log_gf))
        assert gamma_blue > gamma_red, (
            "Blue-ward lines should have larger gamma_rad (∝ λ^{-2})"
        )

    def test_gamma_rad_positive(self, reference_data):
        """gamma_rad must always be positive for valid inputs."""
        from korg.linelist import approximate_radiative_gamma

        for wl in [3e-5, 5e-5, 8e-5]:
            for log_gf in [-3.0, -1.0, 0.0, 1.0]:
                val = float(approximate_radiative_gamma(wl, log_gf))
                assert val > 0, f"gamma_rad={val} for wl={wl}, log_gf={log_gf}"


# ===========================================================================
# TestApproximateGammasPhysics
# ===========================================================================

class TestApproximateGammasPhysics:
    """Physics sanity tests for approximate_gammas."""

    def test_all_reference_cases(self, reference_data):
        """Every case in the JSON must match Julia to rtol=1e-5."""
        from korg.linelist import approximate_gammas
        from korg.species import Species

        if "approximate_gammas" not in reference_data:
            pytest.skip("approximate_gammas reference data not available")

        outputs = reference_data["approximate_gammas"]["outputs"]
        for key, julia_result in outputs.items():
            wl, species_str, E_lower = _parse_approx_gammas_key(key)
            species = Species(species_str)
            gamma_stark, log_gamma_vdW = approximate_gammas(wl, species, E_lower)

            assert np.isclose(float(gamma_stark), julia_result["gamma_stark"], rtol=1e-5), (
                f"gamma_stark mismatch {key}: Python={gamma_stark}, Julia={julia_result['gamma_stark']}"
            )
            assert np.isclose(float(log_gamma_vdW), julia_result["log_gamma_vdW"], rtol=1e-5), (
                f"log_gamma_vdW mismatch {key}: Python={log_gamma_vdW}, Julia={julia_result['log_gamma_vdW']}"
            )

    def test_log_gamma_vdW_is_negative(self, reference_data):
        """log10(gamma_vdW) should be negative for all normal stellar lines."""
        from korg.linelist import approximate_gammas
        from korg.species import Species

        if "approximate_gammas" not in reference_data:
            pytest.skip("approximate_gammas reference data not available")

        outputs = reference_data["approximate_gammas"]["outputs"]
        for key, julia_result in outputs.items():
            log_vdW = julia_result["log_gamma_vdW"]
            assert log_vdW < 0, (
                f"log_gamma_vdW={log_vdW} is positive for key={key}, expected < 0"
            )

    def test_vdW_stronger_for_lower_ionization(self):
        """
        Lower ionization potential → stronger vdW broadening.
        Ca I (IP ~ 6.11 eV) vs Fe I (IP ~ 7.90 eV): Ca I should have
        a larger (less negative) log_gamma_vdW at comparable conditions.
        """
        from korg.linelist import approximate_gammas
        from korg.species import Species

        wl = 5e-5
        E_lower = 0.0

        sp_ca = Species("Ca I")
        _, log_vdW_ca = approximate_gammas(wl, sp_ca, E_lower)

        sp_fe = Species("Fe I")
        _, log_vdW_fe = approximate_gammas(wl, sp_fe, E_lower)

        # Ca has lower IP → upper level closer to ionization limit → larger Δr² → larger gamma
        assert float(log_vdW_ca) > float(log_vdW_fe), (
            f"Expected Ca I log_vdW ({log_vdW_ca:.4f}) > Fe I log_vdW ({log_vdW_fe:.4f})"
        )

    def test_ionized_vs_neutral_stark(self):
        """
        Ionized species should have different Stark broadening to neutral.
        Fe II and Fe I at same conditions should differ.
        """
        from korg.linelist import approximate_gammas
        from korg.species import Species

        wl = 5e-5
        E_lower = 1.0

        sp_fe1 = Species("Fe I")
        gamma_stark_fe1, _ = approximate_gammas(wl, sp_fe1, E_lower)

        sp_fe2 = Species("Fe II")
        gamma_stark_fe2, _ = approximate_gammas(wl, sp_fe2, E_lower)

        # They should not be the same (different Cowley formulas apply)
        assert not np.isclose(float(gamma_stark_fe1), float(gamma_stark_fe2), rtol=0.01), (
            "Fe I and Fe II should have different Stark broadening"
        )

    def test_molecule_returns_zeros(self):
        """Molecules are not handled by the vdW/Stark approximations; result should be 0."""
        from korg.linelist import approximate_gammas
        from korg.species import Species

        sp_cn = Species("CN")
        gamma_stark, log_gamma_vdW = approximate_gammas(5e-5, sp_cn, 0.5)
        assert gamma_stark == 0.0, f"Expected 0.0 for molecule, got {gamma_stark}"
        assert log_gamma_vdW == 0.0, f"Expected 0.0 for molecule, got {log_gamma_vdW}"

    def test_gamma_stark_is_small(self):
        """Stark broadening parameter should be many orders of magnitude below gamma_rad."""
        from korg.linelist import approximate_gammas, approximate_radiative_gamma
        from korg.species import Species

        sp = Species("Fe I")
        wl, E_lower = 5e-5, 1.0
        gamma_stark, _ = approximate_gammas(wl, sp, E_lower)
        gamma_rad = float(approximate_radiative_gamma(wl, -1.0))

        # gamma_stark at 10 000 K is a broadening *coefficient* per electron;
        # the per-electron value is expected to be much smaller than gamma_rad
        assert float(gamma_stark) < gamma_rad, (
            f"Stark coeff ({gamma_stark}) should be smaller than gamma_rad ({gamma_rad})"
        )


# ===========================================================================
# TestLineExplicitBroadeningExtended
# ===========================================================================

class TestLineExplicitBroadeningExtended:
    """Tests for lines created with explicit broadening parameters."""

    def _get_ref(self, reference_data):
        if "line_explicit_broadening" not in reference_data:
            pytest.skip("line_explicit_broadening reference data not available")
        return reference_data["line_explicit_broadening"]

    def test_explicit_gamma_rad_stored_exactly(self, reference_data):
        """Explicit gamma_rad must not be overwritten by approximation."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, test_case in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            wl, log_gf, species_str, E_lower, gamma_rad, gamma_stark, vdW = test_case
            line = create_line(wl, log_gf, species_str, E_lower,
                               gamma_rad=gamma_rad, gamma_stark=gamma_stark, vdW=vdW)
            julia = outputs[key]
            assert np.isclose(line.gamma_rad, julia["gamma_rad"], rtol=1e-6), (
                f"gamma_rad mismatch case {key}: Python={line.gamma_rad}, Julia={julia['gamma_rad']}"
            )

    def test_explicit_gamma_stark_stored_exactly(self, reference_data):
        """Explicit gamma_stark must not be overwritten."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, test_case in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            wl, log_gf, species_str, E_lower, gamma_rad, gamma_stark, vdW = test_case
            line = create_line(wl, log_gf, species_str, E_lower,
                               gamma_rad=gamma_rad, gamma_stark=gamma_stark, vdW=vdW)
            julia = outputs[key]
            assert np.isclose(line.gamma_stark, julia["gamma_stark"], rtol=1e-6), (
                f"gamma_stark mismatch case {key}: Python={line.gamma_stark}, Julia={julia['gamma_stark']}"
            )

    def test_explicit_vdW_stored_correctly(self, reference_data):
        """Explicit vdW [log_gamma, alpha] must be correctly decoded into the tuple."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, test_case in enumerate(inputs):
            key = str(i + 1)
            if key not in outputs:
                continue
            wl, log_gf, species_str, E_lower, gamma_rad, gamma_stark, vdW = test_case
            line = create_line(wl, log_gf, species_str, E_lower,
                               gamma_rad=gamma_rad, gamma_stark=gamma_stark, vdW=vdW)
            julia = outputs[key]
            julia_vdW = julia["vdW"]
            assert np.isclose(line.vdW[0], julia_vdW[0], rtol=1e-5), (
                f"vdW[0] mismatch case {key}: Python={line.vdW[0]}, Julia={julia_vdW[0]}"
            )
            assert np.isclose(line.vdW[1], julia_vdW[1], rtol=1e-10), (
                f"vdW[1] mismatch case {key}: Python={line.vdW[1]}, Julia={julia_vdW[1]}"
            )

    def test_vdW_is_tuple(self, reference_data):
        """line.vdW must be a 2-element tuple."""
        from korg.linelist import create_line

        ref = self._get_ref(reference_data)
        inputs = ref["inputs"]

        for i, test_case in enumerate(inputs):
            wl, log_gf, species_str, E_lower, gamma_rad, gamma_stark, vdW = test_case
            line = create_line(wl, log_gf, species_str, E_lower,
                               gamma_rad=gamma_rad, gamma_stark=gamma_stark, vdW=vdW)
            assert isinstance(line.vdW, tuple), (
                f"vdW should be a tuple, got {type(line.vdW)}"
            )
            assert len(line.vdW) == 2, (
                f"vdW tuple should have 2 elements, got {len(line.vdW)}"
            )

    def test_round_trip_explicit_values(self):
        """Store explicit broadening values then read them back: exact round-trip."""
        from korg.linelist import Line, create_line
        from korg.species import Species

        gamma_rad_in = 1.23456789e8
        gamma_stark_in = 4.56789012e-6
        vdW_in = [-7.777, 0.3]  # as list (like JSON deserialization)

        line = create_line(
            wl=5000.0,
            log_gf=-1.0,
            species="Fe 1",
            E_lower=1.5,
            gamma_rad=gamma_rad_in,
            gamma_stark=gamma_stark_in,
            vdW=vdW_in,
        )

        assert np.isclose(line.gamma_rad, gamma_rad_in, rtol=1e-12)
        assert np.isclose(line.gamma_stark, gamma_stark_in, rtol=1e-12)
        # vdW passed as a 2-element list is normalised to a tuple and stored directly
        # (the negative-log decoding only applies to a scalar vdW, not a pre-formed pair)
        assert line.vdW == tuple(vdW_in), (
            f"Expected vdW={tuple(vdW_in)}, got {line.vdW}"
        )


# ===========================================================================
# TestLinePhysics (non-reference, pure physics)
# ===========================================================================

class TestLinePhysics:
    """Physics sanity checks that do not require Julia reference data."""

    def test_visible_line_wl_in_cm(self):
        """A 5000 Å line should have wl ~ 5e-5 cm."""
        from korg.linelist import create_line

        line = create_line(5000.0, -1.5, "Fe 1", 1.01)
        assert np.isclose(line.wl, 5e-5, rtol=1e-10), (
            f"Expected wl ~ 5e-5 cm, got {line.wl}"
        )

    def test_neutral_species_charge_zero(self):
        """Fe I (neutral) must have charge 0."""
        from korg.linelist import create_line

        line = create_line(5000.0, -1.5, "Fe 1", 1.01)
        assert line.species.charge == 0

    def test_singly_ionized_charge_one(self):
        """Ca II (singly ionized) must have charge 1."""
        from korg.linelist import create_line

        line = create_line(3933.0, 0.135, "Ca 2", 0.0)
        assert line.species.charge == 1

    def test_E_lower_nonnegative(self):
        """Lower energy level must always be >= 0."""
        from korg.linelist import create_line

        for E_lower in [0.0, 0.5, 1.0, 5.0]:
            line = create_line(5000.0, -1.0, "Fe 1", E_lower)
            assert line.E_lower >= 0.0

    def test_gamma_rad_scales_with_gf(self):
        """
        gamma_rad ∝ 10^log_gf — doubling log_gf by 1 dex should give 10× gamma_rad.
        """
        from korg.linelist import approximate_radiative_gamma

        wl = 5e-5
        g1 = float(approximate_radiative_gamma(wl, 0.0))
        g2 = float(approximate_radiative_gamma(wl, 1.0))
        ratio = g2 / g1
        assert np.isclose(ratio, 10.0, rtol=1e-6), (
            f"gamma_rad ratio for Δlog_gf=1 should be 10, got {ratio}"
        )

    def test_line_is_immutable(self):
        """Line is a frozen dataclass — attribute assignment must raise."""
        from korg.linelist import create_line

        line = create_line(5000.0, -1.5, "Fe 1", 1.01)
        with pytest.raises((TypeError, AttributeError)):
            line.wl = 9999.0  # type: ignore[misc]

    def test_Ha_line_wavelength(self):
        """H I 6563 Å line should convert to ~ 6.563e-5 cm."""
        from korg.linelist import create_line

        line = create_line(6563.0, 0.71, "H 1", 10.2)
        assert np.isclose(line.wl, 6.563e-5, rtol=1e-8)
        assert line.species.charge == 0  # H I is neutral

    def test_gamma_rad_inverse_lambda_squared(self):
        """
        gamma_rad ∝ λ^{-2} * gf.
        Comparing two wavelengths at same log_gf should satisfy this.
        """
        from korg.linelist import approximate_radiative_gamma

        log_gf = 0.0
        wl1, wl2 = 4e-5, 8e-5
        g1 = float(approximate_radiative_gamma(wl1, log_gf))
        g2 = float(approximate_radiative_gamma(wl2, log_gf))
        expected_ratio = (wl2 / wl1) ** 2  # g1/g2 should equal (wl2/wl1)^2
        actual_ratio = g1 / g2
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-6), (
            f"Expected lambda^-2 scaling ratio {expected_ratio}, got {actual_ratio}"
        )


# ===========================================================================
# TestJITCompatibility
# ===========================================================================

class TestJITCompatibility:
    """Verify that numeric functions are JIT-compatible."""

    def test_approximate_radiative_gamma_jit(self):
        """approximate_radiative_gamma should JIT-compile and run correctly."""
        from korg.linelist import approximate_radiative_gamma

        @jax.jit
        def jit_func(wl, log_gf):
            return approximate_radiative_gamma(wl, log_gf)

        wl = jnp.float64(5e-5)
        log_gf = jnp.float64(-1.5)
        result_jit = float(jit_func(wl, log_gf))
        result_eager = float(approximate_radiative_gamma(5e-5, -1.5))

        assert np.isclose(result_jit, result_eager, rtol=1e-6), (
            f"JIT result {result_jit} differs from eager {result_eager}"
        )

    def test_approximate_radiative_gamma_jit_array(self):
        """JIT-compiled approximate_radiative_gamma over a wavelength array."""
        from korg.linelist import approximate_radiative_gamma

        @jax.jit
        def batch_func(wls, log_gf):
            return jax.vmap(lambda wl: approximate_radiative_gamma(wl, log_gf))(wls)

        wls = jnp.array([3e-5, 5e-5, 7e-5])
        results = batch_func(wls, jnp.float64(0.0))
        assert results.shape == (3,)
        # Results must be positive and decreasing as wavelength increases
        assert float(results[0]) > float(results[1]) > float(results[2])

    def test_approximate_gammas_numeric_core_jit(self):
        """
        The pure numeric parts of approximate_gammas should be reproducible
        when called repeatedly (a proxy for JIT stability).
        """
        from korg.linelist import approximate_gammas
        from korg.species import Species

        sp = Species("Fe I")
        r1 = approximate_gammas(5e-5, sp, 1.01)
        r2 = approximate_gammas(5e-5, sp, 1.01)
        assert float(r1[0]) == float(r2[0])
        assert float(r1[1]) == float(r2[1])
