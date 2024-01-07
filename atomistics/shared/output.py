"""
The output module defines the abstract output classes for the different types of outputs defined by the atomistics
package. All output classes are abstract classes, which define the output as abstract properties and are derived from
the atomistics.shared.output.AbstractOutput class.
"""

from abc import ABC, abstractmethod

import numpy as np


class AbstractOutput:
    """
    Abstract Base class used for the implementation of the individual output classes.
    """

    def get_output(self, output_keys) -> dict:
        """
        Evaluate multiple properties with a single function call by providing a list of output keys each referencing one
        property as input and returning a dictionary with the property names as keys and the corresponding results as
        values.

        Args:
            output_keys (tuple): Tuple of output property names as strings to be evaluated

        Returns:
            dict: dictionary with the property names as keys and the corresponding results as values.
        """
        return {q: getattr(self, q) for q in output_keys}

    @classmethod
    def keys(cls) -> tuple:
        """
        Return all public functions and properties defined in a given class.

        Returns:
            tuple: Tuple of names all public functions and properties defined in the derived class as strings.
        """
        return tuple(
            [
                k
                for k in cls.__dict__.keys()
                if k[0] != "_" and k not in ["get_output", "keys"]
            ]
        )


class OutputStatic(ABC, AbstractOutput):
    """
    Output class for a static calculation of a supercell with n atoms.
    """

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


class OutputMolecularDynamics(ABC, AbstractOutput):
    """
    Output class for a molecular dynamics calculation with t steps of a supercell with n atoms.
    """

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


class OutputThermalExpansion(ABC, AbstractOutput):
    """
    Output class for a thermal expansion calculation iterating over T temperature steps.
    """

    @property
    @abstractmethod
    def temperatures(self) -> np.ndarray:  # (T) [K]
        pass

    @property
    @abstractmethod
    def volumes(self) -> np.ndarray:  # (T) [Ang^3]
        pass


class OutputThermodynamic(ABC, AbstractOutput):
    """
    Output class for the calculation of the temperature dependence in T temperature steps of thermodynamic properties
    """

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


class OutputEnergyVolumeCurve(ABC, AbstractOutput):
    """
    Output class for the calculation on an energy volume curve calculation based on V strained cells.
    """

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


class OutputElastic(ABC, AbstractOutput):
    """
    Output class for the calculation of elastic moduli from the elastic matrix of the elastic constants.
    """

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


class OutputPhonons(ABC, AbstractOutput):
    """
    Output class for the calculation of phonons using the finite displacement method
    """

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
