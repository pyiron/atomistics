import scipy.constants
from phonopy.units import VaspToTHz, THzToEv, EvTokJmol

kJ_mol_to_eV = 1000 / scipy.constants.Avogadro / scipy.constants.electron_volt
kb = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
