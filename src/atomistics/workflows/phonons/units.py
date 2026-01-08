import scipy.constants
from phonopy.units import EvTokJmol, THzToEv, VaspToTHz

kJ_mol_to_eV = 1000 / scipy.constants.Avogadro / scipy.constants.electron_volt
kb = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

__all__ = ["kb", "kJ_mol_to_eV", "VaspToTHz", "THzToEv", "EvTokJmol"]
