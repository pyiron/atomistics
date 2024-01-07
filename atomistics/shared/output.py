from abc import ABC, abstractmethod

import numpy as np


class Output:
    def get_output(self, output_keys):
        return {q: getattr(self, q) for q in output_keys}

    @classmethod
    def keys(cls):
        return tuple(
            [
                k
                for k in cls.__dict__.keys()
                if k[0] != "_" and k not in ["get_output", "keys"]
            ]
        )


class OutputStatic(ABC, Output):
    @property
    @abstractmethod
    def forces(self) -> np.ndarray:  # (n, 3) [eV / Ang^2]
        pass

    @property
    @abstractmethod
    def energy(self) -> float:  # [eV]
        pass

    @property
    @abstractmethod
    def stress(self) -> np.ndarray:  # (3, 3) [GPa]
        pass

    @property
    @abstractmethod
    def volume(self) -> float:  # [Ang^3]
        pass


class OutputMolecularDynamics(ABC, Output):
    @property
    @abstractmethod
    def positions(self) -> np.ndarray:  # (t, n, 3) [Ang]
        pass

    @property
    @abstractmethod
    def cell(self) -> np.ndarray:  # (t, 3, 3) [Ang]
        pass

    @property
    @abstractmethod
    def forces(self) -> np.ndarray:  # (n, 3) [eV / Ang^2]
        pass

    @property
    @abstractmethod
    def temperature(self) -> np.ndarray:  # (t) [K]
        pass

    @property
    @abstractmethod
    def energy_pot(self) -> np.ndarray:  # (t) [eV]
        pass

    @property
    @abstractmethod
    def energy_tot(self) -> np.ndarray:  # (t) [eV]
        pass

    @property
    @abstractmethod
    def pressure(self) -> np.ndarray:  # (t, 3, 3) [GPa]
        pass

    @property
    @abstractmethod
    def velocities(self) -> np.ndarray:  # (t, n, 3) [eV / Ang]
        pass

    @property
    @abstractmethod
    def volume(self) -> np.ndarray:  # (t) [Ang^3]
        pass


class OutputThermalExpansion(ABC, Output):
    @property
    @abstractmethod
    def temperatures(self) -> np.ndarray:  # (T) [K]
        pass

    @property
    @abstractmethod
    def volumes(self) -> np.ndarray:  # (T) [Ang^3]
        pass


class OutputThermodynamic(ABC, Output):
    @property
    @abstractmethod
    def temperatures(self) -> np.ndarray:  # (T) [K]
        pass

    @property
    @abstractmethod
    def volumes(self) -> np.ndarray:  # (T) [Ang^3]
        pass

    @property
    @abstractmethod
    def free_energy(self) -> np.ndarray:  # (T) [eV]
        pass

    @property
    @abstractmethod
    def entropy(self) -> np.ndarray:  # (T) [eV]
        pass

    @property
    @abstractmethod
    def heat_capacity(self) -> np.ndarray:  # (T) [eV]
        pass


class OutputEnergyVolumeCurve(ABC, Output):
    @property
    @abstractmethod
    def energy_eq(self) -> float:  # float [eV]
        pass

    @property
    @abstractmethod
    def volume_eq(self) -> float:  # float [Ang^3]
        pass

    @property
    @abstractmethod
    def bulkmodul_eq(self) -> float:  # float [GPa]
        pass

    @property
    @abstractmethod
    def b_prime_eq(self) -> float:  # float
        pass

    @property
    @abstractmethod
    def fit_dict(self) -> dict:  # dict
        pass

    @property
    @abstractmethod
    def energy(self) -> np.ndarray:  # (V) [eV]
        pass

    @property
    @abstractmethod
    def volume(self) -> np.ndarray:  # (V) [Ang^3]
        pass


class OutputElastic(ABC, Output):
    @property
    @abstractmethod
    def elastic_matrix(self) -> np.ndarray:  # (6,6) [GPa]
        pass

    @property
    @abstractmethod
    def elastic_matrix_inverse(self) -> np.ndarray:  # (6,6) [GPa]
        pass

    @property
    @abstractmethod
    def bulkmodul_voigt(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def bulkmodul_reuss(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def bulkmodul_hill(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def shearmodul_voigt(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def shearmodul_reuss(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def shearmodul_hill(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def youngsmodul_voigt(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def youngsmodul_reuss(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def youngsmodul_hill(self) -> float:  # [GPa]
        pass

    @property
    @abstractmethod
    def poissonsratio_voigt(self) -> float:
        pass

    @property
    @abstractmethod
    def poissonsratio_reuss(self) -> float:
        pass

    @property
    @abstractmethod
    def poissonsratio_hill(self) -> float:
        pass

    @property
    @abstractmethod
    def AVR(self) -> float:
        pass

    @property
    @abstractmethod
    def elastic_matrix_eigval(self) -> np.ndarray:  # (6,6) [GPa]
        pass


class OutputPhonons(ABC, Output):
    @property
    @abstractmethod
    def mesh_dict(self) -> dict:
        pass

    @property
    @abstractmethod
    def band_structure_dict(self) -> dict:
        pass

    @property
    @abstractmethod
    def total_dos_dict(self) -> dict:
        pass

    @property
    @abstractmethod
    def dynamical_matrix(self) -> dict:
        pass

    @property
    @abstractmethod
    def force_constants(self) -> dict:
        pass
