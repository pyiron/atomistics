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
    """Output data for a static (single-point) calculation.

    Attributes:
        forces (OutputCallable): Callable returning atomic forces in eV/Å.
        energy (OutputCallable): Callable returning the total energy in eV.
        stress (OutputCallable): Callable returning the stress tensor in eV/Å³.
        volume (OutputCallable): Callable returning the cell volume in Å³.
    """

    forces: OutputCallable
    energy: OutputCallable
    stress: OutputCallable
    volume: OutputCallable


@dataclasses.dataclass
class OutputMolecularDynamics(Output):
    """Output data for a molecular dynamics calculation.

    Attributes:
        positions (OutputCallable): Callable returning atomic positions in Å.
        cell (OutputCallable): Callable returning the simulation cell matrix in Å.
        forces (OutputCallable): Callable returning atomic forces in eV/Å.
        temperature (OutputCallable): Callable returning the instantaneous temperature in K.
        energy_pot (OutputCallable): Callable returning the potential energy in eV.
        energy_tot (OutputCallable): Callable returning the total energy in eV.
        pressure (OutputCallable): Callable returning the pressure tensor in GPa.
        velocities (OutputCallable): Callable returning atomic velocities in Å/ps.
        volume (OutputCallable): Callable returning the cell volume in Å³.
    """

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
    """Output data for a thermal expansion calculation.

    Attributes:
        temperatures (OutputCallable): Callable returning the list of temperatures in K.
        volumes (OutputCallable): Callable returning the equilibrium volumes at each temperature in Å³.
    """

    temperatures: OutputCallable
    volumes: OutputCallable


@dataclasses.dataclass
class OutputThermodynamic(OutputThermalExpansion):
    """Output data for thermodynamic properties derived from thermal expansion.

    Extends OutputThermalExpansion with thermodynamic quantities.

    Attributes:
        free_energy (OutputCallable): Callable returning the Helmholtz free energy in eV.
        entropy (OutputCallable): Callable returning the entropy in eV/K.
        heat_capacity (OutputCallable): Callable returning the heat capacity in eV/K.
    """

    free_energy: OutputCallable
    entropy: OutputCallable
    heat_capacity: OutputCallable


@dataclasses.dataclass
class EquilibriumEnergy(Output):
    """Output data for the equilibrium energy from an equation-of-state fit.

    Attributes:
        energy_eq (OutputCallable): Callable returning the equilibrium energy in eV.
    """

    energy_eq: OutputCallable


@dataclasses.dataclass
class EquilibriumVolume(Output):
    """Output data for the equilibrium volume from an equation-of-state fit.

    Attributes:
        volume_eq (OutputCallable): Callable returning the equilibrium volume in Å³.
    """

    volume_eq: OutputCallable


@dataclasses.dataclass
class EquilibriumBulkModul(Output):
    """Output data for the equilibrium bulk modulus from an equation-of-state fit.

    Attributes:
        bulkmodul_eq (OutputCallable): Callable returning the equilibrium bulk modulus in GPa.
    """

    bulkmodul_eq: OutputCallable


@dataclasses.dataclass
class EquilibriumBulkModulDerivative(Output):
    """Output data for the pressure derivative of the bulk modulus from an EOS fit.

    Attributes:
        b_prime_eq (OutputCallable): Callable returning the equilibrium bulk modulus pressure derivative (dimensionless).
    """

    b_prime_eq: OutputCallable


@dataclasses.dataclass
class OutputEnergyVolumeCurve(
    EquilibriumEnergy,
    EquilibriumVolume,
    EquilibriumBulkModul,
    EquilibriumBulkModulDerivative,
):
    """Output data for an energy-volume curve workflow.

    Combines equilibrium properties with the raw EOS fit data.

    Attributes:
        fit_dict (OutputCallable): Callable returning the full EOS fit parameter dictionary.
        energy (OutputCallable): Callable returning the energies at each sampled volume in eV.
        volume (OutputCallable): Callable returning the sampled volumes in Å³.
    """

    fit_dict: OutputCallable
    energy: OutputCallable
    volume: OutputCallable


@dataclasses.dataclass
class OutputElastic(Output):
    """Output data for an elastic constants calculation.

    All moduli are in GPa; Poisson ratios and AVR are dimensionless.

    Attributes:
        elastic_matrix (OutputCallable): Callable returning the 6×6 Voigt elastic matrix in GPa.
        elastic_matrix_inverse (OutputCallable): Callable returning the compliance matrix (inverse of elastic_matrix).
        bulkmodul_voigt (OutputCallable): Callable returning the Voigt-averaged bulk modulus in GPa.
        bulkmodul_reuss (OutputCallable): Callable returning the Reuss-averaged bulk modulus in GPa.
        bulkmodul_hill (OutputCallable): Callable returning the Hill-averaged bulk modulus in GPa.
        shearmodul_voigt (OutputCallable): Callable returning the Voigt-averaged shear modulus in GPa.
        shearmodul_reuss (OutputCallable): Callable returning the Reuss-averaged shear modulus in GPa.
        shearmodul_hill (OutputCallable): Callable returning the Hill-averaged shear modulus in GPa.
        youngsmodul_voigt (OutputCallable): Callable returning the Voigt-averaged Young's modulus in GPa.
        youngsmodul_reuss (OutputCallable): Callable returning the Reuss-averaged Young's modulus in GPa.
        youngsmodul_hill (OutputCallable): Callable returning the Hill-averaged Young's modulus in GPa.
        poissonsratio_voigt (OutputCallable): Callable returning the Voigt-averaged Poisson's ratio.
        poissonsratio_reuss (OutputCallable): Callable returning the Reuss-averaged Poisson's ratio.
        poissonsratio_hill (OutputCallable): Callable returning the Hill-averaged Poisson's ratio.
        AVR (OutputCallable): Callable returning the Voigt-Reuss-Hill anisotropy ratio.
        elastic_matrix_eigval (OutputCallable): Callable returning the eigenvalues of the elastic matrix.
    """

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
    """Output data for a phonon calculation.

    Attributes:
        mesh_dict (OutputCallable): Callable returning the phonon mesh sampling results.
        band_structure_dict (OutputCallable): Callable returning the phonon band structure along high-symmetry paths.
        total_dos_dict (OutputCallable): Callable returning the total phonon density of states.
        dynamical_matrix (OutputCallable): Callable returning the dynamical matrix.
        force_constants (OutputCallable): Callable returning the interatomic force constants matrix.
    """

    mesh_dict: OutputCallable
    band_structure_dict: OutputCallable
    total_dos_dict: OutputCallable
    dynamical_matrix: OutputCallable
    force_constants: OutputCallable
