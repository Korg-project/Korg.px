"""
Species and Formula types for representing atoms and molecules.
"""

import re
import numpy as np
from typing import Union, List
from dataclasses import dataclass
from .atomic_data import atomic_symbols, atomic_numbers, atomic_masses, MAX_ATOMIC_NUMBER

MAX_ATOMS_PER_MOLECULE = 6

# Matches Korg.jl's `roman_numerals` in species.jl.
ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

# Julia: split(code, [' ', '.', '_']).  "-" can't be a separator (minus signs).
_SPECIES_SEPARATORS = re.compile(r"[ ._]")

@dataclass(frozen=True)
class Formula:
    """
    Represents an atom or molecule, irrespective of its charge.

    Internally stores atoms as a sorted array of atomic numbers.
    Supports up to MAX_ATOMS_PER_MOLECULE (6) atoms.
    """
    atoms: tuple  # Tuple of 6 uint8 values, padded with zeros

    def __init__(self, atoms_input: Union[int, List[int], str, 'Formula']):
        """
        Construct a Formula from various input types:
        - int: single atomic number (e.g., 26 for Fe)
        - list: list of atomic numbers (e.g., [1, 8] for OH)
        - str: atomic symbol, molecule, or numeric code (e.g., "Fe", "OH", "FeH", "0801")
        - Formula: copy constructor
        """
        if isinstance(atoms_input, Formula):
            object.__setattr__(self, 'atoms', atoms_input.atoms)
            return

        if isinstance(atoms_input, str):
            atoms_list = self._parse_string(atoms_input)
        elif isinstance(atoms_input, int):
            if not 1 <= atoms_input <= MAX_ATOMIC_NUMBER:
                raise ValueError(f"Atomic number must be between 1 and {MAX_ATOMIC_NUMBER}")
            atoms_list = [atoms_input]
        elif isinstance(atoms_input, (list, tuple)):
            if len(atoms_input) == 0:
                raise ValueError("Can't construct an empty Formula")
            if len(atoms_input) > MAX_ATOMS_PER_MOLECULE:
                raise ValueError(f"Can't construct Formula with {len(atoms_input)} atoms. "
                               f"Up to {MAX_ATOMS_PER_MOLECULE} atoms are supported.")
            atoms_list = list(atoms_input)
            if not all(1 <= a <= MAX_ATOMIC_NUMBER for a in atoms_list):
                raise ValueError("All atomic numbers must be between 1 and {MAX_ATOMIC_NUMBER}")
        else:
            raise TypeError(f"Invalid input type for Formula: {type(atoms_input)}")

        # Sort and pad to MAX_ATOMS_PER_MOLECULE
        atoms_list = sorted(atoms_list)
        padded = [0] * (MAX_ATOMS_PER_MOLECULE - len(atoms_list)) + atoms_list
        object.__setattr__(self, 'atoms', tuple(np.uint8(padded)))

    @classmethod
    def from_string(cls, code: str) -> 'Formula':
        """Construct a Formula from a string code (e.g. 'Fe', 'OH', 'FeH', '0801')."""
        return cls(code)

    def _parse_string(self, code: str) -> List[int]:
        """Parse a string code into a list of atomic numbers."""
        # Quick parse for single elements
        if code in atomic_symbols:
            return [atomic_numbers[code]]

        # Handle numeric codes (e.g., "0801" -> OH, "26" -> Fe)
        if all(c.isdigit() for c in code):
            if len(code) <= 2:
                return [int(code)]
            elif len(code) <= 4:
                el1 = int(code[:len(code)-2])
                el2 = int(code[-2:])
                return sorted([el1, el2])
            else:
                # Pad with leading zero if odd length
                if len(code) % 2 == 1:
                    code = "0" + code
                els = [int(code[i:i+2]) for i in range(0, len(code), 2)]
                if len(els) > MAX_ATOMS_PER_MOLECULE:
                    raise ValueError(f"Korg only supports atoms with up to "
                                   f"{MAX_ATOMS_PER_MOLECULE} nuclei. (Trying to parse {code})")
                return sorted(els)

        # Parse molecular formulas like "OH", "FeH", "C2", "H2O"
        atoms_list = []
        i = 0
        while i < len(code):
            if code[i].isupper():
                # Start of element symbol
                symbol = code[i]
                i += 1
                # Check for lowercase letter (e.g., "Fe")
                if i < len(code) and code[i].islower():
                    symbol += code[i]
                    i += 1

                if symbol not in atomic_numbers:
                    raise ValueError(f"Unknown element symbol: {symbol}")

                # Check for number (e.g., "H2")
                count = 1
                if i < len(code) and code[i].isdigit():
                    num_str = ""
                    while i < len(code) and code[i].isdigit():
                        num_str += code[i]
                        i += 1
                    count = int(num_str)

                atoms_list.extend([atomic_numbers[symbol]] * count)
            else:
                i += 1

        if not atoms_list:
            raise ValueError(f"Could not parse formula: {code}")

        return sorted(atoms_list)

    def get_atoms(self) -> np.ndarray:
        """Returns array view of the atomic numbers (excluding padding zeros)."""
        atoms_arr = np.array(self.atoms, dtype=np.uint8)
        first_nonzero = np.argmax(atoms_arr > 0)
        if atoms_arr[first_nonzero] == 0:
            # All zeros - should not happen with valid Formula
            return np.array([], dtype=np.uint8)
        return atoms_arr[first_nonzero:]

    def get_atom(self) -> int:
        """Returns the atomic number (only valid for single atoms, not molecules).

        The result is a Python ``int``, not a ``np.uint8``: the atom vector is
        stored as uint8, and under NEP 50 an expression like ``Z * 100`` on a
        uint8 wraps around silently.
        """
        if self.is_molecule():
            raise ValueError("Can't get the atomic number of a molecule. Use get_atoms() instead.")
        return int(self.get_atoms()[0])

    def n_atoms(self) -> int:
        """Returns the number of atoms in the formula."""
        return len(self.get_atoms())

    def is_molecule(self) -> bool:
        """Returns True if this formula represents a molecule (more than one atom)."""
        return bool(self.atoms[MAX_ATOMS_PER_MOLECULE - 2] != 0)

    def get_mass(self) -> float:
        """Returns the mass in grams."""
        return sum(atomic_masses[a - 1] for a in self.get_atoms())

    def __str__(self) -> str:
        """String representation (e.g., 'Fe', 'OH', 'H2O')."""
        atoms_arr = self.get_atoms()
        if len(atoms_arr) == 0:
            return ""

        result = []
        i = 0
        while i < len(atoms_arr):
            atom = atoms_arr[i]
            count = 1
            while i + count < len(atoms_arr) and atoms_arr[i + count] == atom:
                count += 1

            symbol = atomic_symbols[atom - 1]
            if count == 1:
                result.append(symbol)
            else:
                result.append(f"{symbol}{count}")
            i += count

        return "".join(result)

    def __repr__(self) -> str:
        return f"Formula('{str(self)}')"

    def __hash__(self):
        return hash(self.atoms)

    def __eq__(self, other):
        if not isinstance(other, Formula):
            return False
        return self.atoms == other.atoms

    def __lt__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return self.atoms < other.atoms

    def __le__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return self.atoms <= other.atoms

    def __gt__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return self.atoms > other.atoms

    def __ge__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return self.atoms >= other.atoms


@dataclass(frozen=True)
class Species:
    """
    Represents an atom or molecule with a particular charge state.

    Examples:
        Species(Formula(26), 0)  -> Fe I   (neutral iron)
        Species(Formula(26), 1)  -> Fe II  (singly ionized iron)
        Species("Fe I")          -> Fe I
        Species("H2O")           -> H2O    (neutral water)
    """
    formula: Formula
    charge: int

    def __init__(self, formula_input: Union[Formula, str, int, float], charge: int = None):
        """
        Construct a Species from various input types.

        Args:
            formula_input: Formula, string code, or atomic number
            charge: charge state (0=neutral, 1=singly ionized, etc.)
                   If None and formula_input is a string, charge is parsed from string
        """
        if isinstance(formula_input, Species):
            object.__setattr__(self, 'formula', formula_input.formula)
            object.__setattr__(self, 'charge', formula_input.charge)
            return

        if isinstance(formula_input, (str, float)):
            # Parse species code
            formula, parsed_charge = self._parse_species_code(str(formula_input))
            charge = parsed_charge if charge is None else int(charge)
            if charge < -1:
                raise ValueError(f"Can't construct a species with charge < -1: "
                                 f"{formula} with charge {charge}")
            object.__setattr__(self, 'formula', formula)
            object.__setattr__(self, 'charge', charge)
        else:
            # Direct construction
            if not isinstance(formula_input, Formula):
                formula_input = Formula(formula_input)
            if charge is None:
                charge = 0
            if charge < -1:
                raise ValueError(f"Can't construct a species with charge < -1: {formula_input} with charge {charge}")
            object.__setattr__(self, 'formula', formula_input)
            object.__setattr__(self, 'charge', int(charge))

    @classmethod
    def from_string(cls, code: str) -> 'Species':
        """Construct a Species from a string code (e.g. 'Fe I', 'H2O', 'OH+')."""
        return cls(code)

    def _parse_species_code(self, code: str) -> tuple:
        """
        Parse a species code string into (Formula, charge).

        This mirrors ``Korg.Species(code::AbstractString)`` in Korg.jl's
        ``species.jl`` statement for statement.
        """
        # Julia: strip(code, ['0', ' ']) -- leading *and trailing* 0s and spaces
        # are safe to remove. This is what makes "02.1000" parse as He II.
        code = code.strip(' 0')

        # If the species ends in "+" or "-", convert it to a numerical charge.
        # The ionization number is charge+1, so "H 0" is H⁻ and "H 2" is H⁺.
        if code.endswith('+'):
            code = code[:-1] + " 2"
        elif code.endswith('-'):
            code = code[:-1] + " 0"

        # Valid separators between the formula part and the charge part.
        # Julia splits on all of them at once and drops empty tokens, so
        # "01.01" parses the same as ".01..01.".
        parts = [p for p in _SPECIES_SEPARATORS.split(code) if p]

        if len(parts) == 0:
            raise ValueError(f"{code!r} isn't a valid species code")
        if len(parts) > 2:
            raise ValueError(f"{code} isn't a valid species code")

        formula = Formula(parts[0])

        if len(parts) == 1:
            charge = 0
        else:
            # Parse charge
            charge_str = parts[1]
            # Check for Roman numerals
            if charge_str in ROMAN_NUMERALS:
                charge = ROMAN_NUMERALS.index(charge_str)
            else:
                charge = int(charge_str)
                # If this is a Kurucz-style numeric code the charge is already
                # correct, otherwise (spectroscopic notation) subtract 1.
                try:
                    float(code)
                except ValueError:
                    charge -= 1

        return formula, charge

    def get_atoms(self) -> np.ndarray:
        """Returns array of atomic numbers."""
        return self.formula.get_atoms()

    def get_atom(self) -> int:
        """Returns the atomic number (only for atomic species)."""
        return self.formula.get_atom()

    def n_atoms(self) -> int:
        """Returns the number of atoms."""
        return self.formula.n_atoms()

    def is_molecule(self) -> bool:
        """Returns True if this is a molecular species."""
        return self.formula.is_molecule()

    def get_mass(self) -> float:
        """Returns the mass in grams."""
        return self.formula.get_mass()

    def __str__(self) -> str:
        """String representation (e.g., 'Fe I', 'Fe II', 'H2O', 'OH+')."""
        formula_str = str(self.formula)

        if self.is_molecule() and self.charge == 1:
            return formula_str + "+"
        elif self.is_molecule() and self.charge == 0:
            return formula_str
        elif self.charge == -1:
            return formula_str + "-"
        elif 0 <= self.charge <= len(ROMAN_NUMERALS) - 1:
            return f"{formula_str} {ROMAN_NUMERALS[self.charge]}"
        else:
            return f"{formula_str} {self.charge}"

    def __repr__(self) -> str:
        return f"Species('{str(self)}')"

    def __hash__(self):
        return hash((self.formula, self.charge))

    def __eq__(self, other):
        if not isinstance(other, Species):
            return False
        return self.formula == other.formula and self.charge == other.charge

    def __lt__(self, other):
        if not isinstance(other, Species):
            return NotImplemented
        return (self.formula, self.charge) < (other.formula, other.charge)

    def __le__(self, other):
        if not isinstance(other, Species):
            return NotImplemented
        return (self.formula, self.charge) <= (other.formula, other.charge)

    def __gt__(self, other):
        if not isinstance(other, Species):
            return NotImplemented
        return (self.formula, self.charge) > (other.formula, other.charge)

    def __ge__(self, other):
        if not isinstance(other, Species):
            return NotImplemented
        return (self.formula, self.charge) >= (other.formula, other.charge)


def all_atomic_species():
    """Returns an iterator over all atomic species supported by Korg."""
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        for charge in range(min(3, Z + 1)):  # 0, 1, 2 or fewer if Z is small
            yield Species(Formula(Z), charge)
