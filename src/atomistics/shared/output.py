from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

OutputFunction = Callable[[], Any]


@dataclasses.dataclass
class Output:
    """
    Base class for output data.
    """

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """
        Get the keys of the output data class.

        Returns:
            tuple: The keys of the output data class.
        """
        return tuple(field.name for field in dataclasses.fields(cls))

    def get(self, output_keys: Iterable[str]) -> dict[str, Any]:
        """
        Get the specified output data.

        Args:
            output_keys (tuple): The keys of the output data to retrieve.

        Returns:
            dict: The output data.
        """
        return {q: getattr(self, q)() for q in output_keys}


@dataclasses.dataclass
class OutputStatic(Output):
    forces: OutputFunction
    energy: OutputFunction
    stress: OutputFunction
    volume: OutputFunction


@dataclasses.dataclass
class OutputMolecularDynamics(Output):
    positions: OutputFunction
    cell: OutputFunction
    forces: OutputFunction
    temperature: OutputFunction
    energy_pot: OutputFunction
    energy_tot: OutputFunction
    pressure: OutputFunction
    velocities: OutputFunction
    volume: OutputFunction


@dataclasses.dataclass
class OutputThermalExpansion(Output):
    temperatures: OutputFunction
    volumes: OutputFunction


@dataclasses.dataclass
class OutputThermodynamic(OutputThermalExpansion):
    free_energy: OutputFunction
    entropy: OutputFunction
    heat_capacity: OutputFunction


@dataclasses.dataclass
class EquilibriumEnergy(Output):
    energy_eq: OutputFunction


@dataclasses.dataclass
class EquilibriumVolume(Output):
    volume_eq: OutputFunction


@dataclasses.dataclass
class EquilibriumBulkModul(Output):
    bulkmodul_eq: OutputFunction


@dataclasses.dataclass
class EquilibriumBulkModulDerivative(Output):
    b_prime_eq: OutputFunction


@dataclasses.dataclass
class OutputEnergyVolumeCurve(
    EquilibriumEnergy,
    EquilibriumVolume,
    EquilibriumBulkModul,
    EquilibriumBulkModulDerivative,
):
    fit_dict: OutputFunction
    energy: OutputFunction
    volume: OutputFunction


@dataclasses.dataclass
class OutputElastic(Output):
    elastic_matrix: OutputFunction
    elastic_matrix_inverse: OutputFunction
    bulkmodul_voigt: OutputFunction
    bulkmodul_reuss: OutputFunction
    bulkmodul_hill: OutputFunction
    shearmodul_voigt: OutputFunction
    shearmodul_reuss: OutputFunction
    shearmodul_hill: OutputFunction
    youngsmodul_voigt: OutputFunction
    youngsmodul_reuss: OutputFunction
    youngsmodul_hill: OutputFunction
    poissonsratio_voigt: OutputFunction
    poissonsratio_reuss: OutputFunction
    poissonsratio_hill: OutputFunction
    AVR: OutputFunction
    elastic_matrix_eigval: OutputFunction


@dataclasses.dataclass
class OutputPhonons(Output):
    mesh_dict: OutputFunction
    band_structure_dict: OutputFunction
    total_dos_dict: OutputFunction
    dynamical_matrix: OutputFunction
    force_constants: OutputFunction
