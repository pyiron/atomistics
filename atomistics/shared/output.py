import dataclasses


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

    def get(self, output_keys: tuple) -> dict:
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
    forces: callable
    energy: callable
    stress: callable
    volume: callable


@dataclasses.dataclass
class OutputMolecularDynamics(Output):
    positions: callable
    cell: callable
    forces: callable
    temperature: callable
    energy_pot: callable
    energy_tot: callable
    pressure: callable
    velocities: callable
    volume: callable


@dataclasses.dataclass
class OutputThermalExpansion(Output):
    temperatures: callable
    volumes: callable


@dataclasses.dataclass
class OutputThermodynamic(OutputThermalExpansion):
    free_energy: callable
    entropy: callable
    heat_capacity: callable


@dataclasses.dataclass
class EquilibriumEnergy(Output):
    energy_eq: callable


@dataclasses.dataclass
class EquilibriumVolume(Output):
    volume_eq: callable


@dataclasses.dataclass
class EquilibriumBulkModul(Output):
    bulkmodul_eq: callable


@dataclasses.dataclass
class EquilibriumBulkModulDerivative(Output):
    b_prime_eq: callable


@dataclasses.dataclass
class OutputEnergyVolumeCurve(
    EquilibriumEnergy,
    EquilibriumVolume,
    EquilibriumBulkModul,
    EquilibriumBulkModulDerivative,
):
    fit_dict: callable
    energy: callable
    volume: callable


@dataclasses.dataclass
class OutputElastic(Output):
    elastic_matrix: callable
    elastic_matrix_inverse: callable
    bulkmodul_voigt: callable
    bulkmodul_reuss: callable
    bulkmodul_hill: callable
    shearmodul_voigt: callable
    shearmodul_reuss: callable
    shearmodul_hill: callable
    youngsmodul_voigt: callable
    youngsmodul_reuss: callable
    youngsmodul_hill: callable
    poissonsratio_voigt: callable
    poissonsratio_reuss: callable
    poissonsratio_hill: callable
    AVR: callable
    elastic_matrix_eigval: callable


@dataclasses.dataclass
class OutputPhonons(Output):
    mesh_dict: callable
    band_structure_dict: callable
    total_dos_dict: callable
    dynamical_matrix: callable
    force_constants: callable
