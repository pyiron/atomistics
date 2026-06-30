import scipy.constants
from phonopy.physical_units import get_physical_units

kJ_mol_to_eV = 1000 / scipy.constants.Avogadro / scipy.constants.electron_volt
kb = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
_phonopy_units = get_physical_units()
THzToEv = _phonopy_units.THzToEv
EvTokJmol = _phonopy_units.EvTokJmol

__all__ = ["kb", "kJ_mol_to_eV", "THzToEv", "EvTokJmol"]
