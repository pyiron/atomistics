import dataclasses
from collections.abc import Callable, Iterable

OutputCallable = Callable[[], object]


@dataclasses.dataclass
class Output:
    """
    Base class for output data.
    """

    @classmethod
    def keys(cls) -> tuple:
        """
        Get the keys of the output data class.

        Returns:
            tuple: The keys of the output data class.
        """
        return tuple(field.name for field in dataclasses.fields(cls))

    def get(self, output_keys: Iterable[str]) -> dict:
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
    forces: OutputCallable
    energy: OutputCallable
    stress: OutputCallable
    volume: OutputCallable


@dataclasses.dataclass
class OutputMolecularDynamics(Output):
    positions: OutputCallable
    cell: OutputCallable
    forces: OutputCallable
    temperature: OutputCallable
    energy_pot: OutputCallable
    energy_tot: OutputCallable
    pressure: OutputCallable
    velocities: OutputCallable
    volume: OutputCallable


@dataclasses.dataclass
class OutputThermalExpansion(Output):
    temperatures: OutputCallable
    volumes: OutputCallable


@dataclasses.dataclass
class OutputThermodynamic(OutputThermalExpansion):
    free_energy: OutputCallable
    entropy: OutputCallable
    heat_capacity: OutputCallable


@dataclasses.dataclass
class EquilibriumEnergy(Output):
    energy_eq: OutputCallable


@dataclasses.dataclass
class EquilibriumVolume(Output):
    volume_eq: OutputCallable


@dataclasses.dataclass
class EquilibriumBulkModul(Output):
    bulkmodul_eq: OutputCallable


@dataclasses.dataclass
class EquilibriumBulkModulDerivative(Output):
    b_prime_eq: OutputCallable


@dataclasses.dataclass
class OutputEnergyVolumeCurve(
    EquilibriumEnergy,
    EquilibriumVolume,
    EquilibriumBulkModul,
    EquilibriumBulkModulDerivative,
):
    fit_dict: OutputCallable
    energy: OutputCallable
    volume: OutputCallable


@dataclasses.dataclass
class OutputElastic(Output):
    elastic_matrix: OutputCallable
    elastic_matrix_inverse: OutputCallable
    bulkmodul_voigt: OutputCallable
    bulkmodul_reuss: OutputCallable
    bulkmodul_hill: OutputCallable
    shearmodul_voigt: OutputCallable
    shearmodul_reuss: OutputCallable
    shearmodul_hill: OutputCallable
    youngsmodul_voigt: OutputCallable
    youngsmodul_reuss: OutputCallable
    youngsmodul_hill: OutputCallable
    poissonsratio_voigt: OutputCallable
    poissonsratio_reuss: OutputCallable
    poissonsratio_hill: OutputCallable
    AVR: OutputCallable
    elastic_matrix_eigval: OutputCallable


@dataclasses.dataclass
class OutputPhonons(Output):
    mesh_dict: OutputCallable
    band_structure_dict: OutputCallable
    total_dos_dict: OutputCallable
    dynamical_matrix: OutputCallable
    force_constants: OutputCallable
