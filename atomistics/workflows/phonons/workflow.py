from typing import Optional
import posixpath

from ase.atoms import Atoms
import numpy as np
from phonopy.file_IO import write_FORCE_CONSTANTS

from atomistics.shared.output import OutputThermodynamic, OutputPhonons
from atomistics.workflows.interface import Workflow
from atomistics.workflows.phonons.helper import (
    analyse_structures_helper,
    generate_structures_helper,
    get_thermal_properties,
    get_hesse_matrix,
    get_band_structure,
    plot_band_structure,
    plot_dos,
)
from atomistics.workflows.phonons.units import VaspToTHz


class PhonopyWorkflow(Workflow):
    """
    Phonopy wrapper for the calculation of free energy in the framework of quasi harmonic approximation.

    Get output via `get_thermal_properties()`.

    Note:

    - This class does not consider the thermal expansion. For this, use `QuasiHarmonicJob` (find more in its
        docstring)
    - Depending on the value given in `job.input['interaction_range']`, this class automatically changes the number of
        atoms. The input parameters of the reference job might have to be set appropriately (e.g. use `k_mesh_density`
        for DFT instead of setting k-points directly).
    - The structure used in the reference job should be a relaxed structure.
    - Theory behind it: https://en.wikipedia.org/wiki/Quasi-harmonic_approximation
    """

    def __init__(
        self,
        structure: Atoms,
        interaction_range: float = 10.0,
        factor: float = VaspToTHz,
        displacement: float = 0.01,
        dos_mesh: int = 20,
        primitive_matrix: np.ndarray = None,
        number_of_snapshots: int = None,
    ):
        self._interaction_range = interaction_range
        self._displacement = displacement
        self._dos_mesh = dos_mesh
        self._number_of_snapshots = number_of_snapshots
        self.structure = structure
        self._primitive_matrix = primitive_matrix
        self._factor = factor
        self.phonopy = None
        self._phonopy_dict = {}

    def generate_structures(self) -> dict:
        self.phonopy, structure_dict = generate_structures_helper(
            structure=self.structure,
            primitive_matrix=self._primitive_matrix,
            displacement=self._displacement,
            number_of_snapshots=self._number_of_snapshots,
            interaction_range=self._interaction_range,
            factor=self._factor,
        )
        return {"calc_forces": structure_dict}

    def analyse_structures(
        self, output_dict: dict, output_keys: tuple[str] = OutputPhonons.keys()
    ) -> dict:
        """

        Returns:

        """
        self._phonopy_dict = analyse_structures_helper(
            phonopy=self.phonopy,
            output_dict=output_dict,
            dos_mesh=self._dos_mesh,
            number_of_snapshots=self._number_of_snapshots,
            output_keys=output_keys,
        )
        return self._phonopy_dict

    def get_thermal_properties(
        self,
        t_min: float = 1.0,
        t_max: float = 1500.0,
        t_step: float = 50.0,
        temperatures: np.ndarray = None,
        cutoff_frequency: float = None,
        pretend_real: bool = False,
        band_indices: np.ndarray = None,
        is_projection: bool = False,
        output_keys: tuple[str] = OutputThermodynamic.keys(),
    ) -> dict:
        """
        Returns thermal properties at constant volume in the given temperature range.  Can only be called after job
        successfully ran.

        Args:
            t_min (float): minimum sample temperature
            t_max (float): maximum sample temperature
            t_step (int):  tempeature sample interval
            temperatures (array_like, float):  custom array of temperature samples, if given t_min, t_max, t_step are
                                               ignored.

        Returns:
            :class:`Thermal`: thermal properties as returned by Phonopy
        """
        return get_thermal_properties(
            phonopy=self.phonopy,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            cutoff_frequency=cutoff_frequency,
            pretend_real=pretend_real,
            band_indices=band_indices,
            is_projection=is_projection,
            output_keys=output_keys,
        )

    def get_dynamical_matrix(self, npoints: int = 101) -> np.ndarray:
        """

        Returns:

        """
        self.phonopy.auto_band_structure(
            npoints=npoints,
            with_eigenvectors=False,
            with_group_velocities=False,
            plot=False,
            write_yaml=False,
            filename="band.yaml",
        )
        return np.real_if_close(self.phonopy.dynamical_matrix.dynamical_matrix)

    def dynamical_matrix_at_q(self, q: np.ndarray) -> np.ndarray:
        """

        Args:
            q:

        Returns:

        """
        return np.real_if_close(self.phonopy.get_dynamical_matrix_at_q(q))

    def write_phonopy_force_constants(
        self, file_name: str = "FORCE_CONSTANTS", cwd: str = None
    ):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)
        write_FORCE_CONSTANTS(
            force_constants=self.phonopy.force_constants, filename=file_name
        )

    def get_hesse_matrix(self) -> np.ndarray:
        return get_hesse_matrix(force_constants=self.phonopy.force_constants)

    def get_band_structure(
        self,
        npoints: int = 101,
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False,
    ):
        return get_band_structure(
            phonopy=self.phonopy,
            npoints=npoints,
            with_eigenvectors=with_eigenvectors,
            with_group_velocities=with_group_velocities,
        )

    def plot_band_structure(
        self, axis=None, *args, label: Optional[str] = None, **kwargs
    ):
        try:
            results = self.phonopy.get_band_structure_dict()
        except RuntimeError:
            results = self.get_band_structure()

        # HACK: strictly speaking this breaks phonopy API and could bite us
        path_connections = self.phonopy._band_structure.path_connections
        labels = self.phonopy._band_structure.labels
        return plot_band_structure(
            results=results,
            path_connections=path_connections,
            labels=labels,
            axis=axis,
            *args,
            label=label,
            **kwargs,
        )

    def plot_dos(self, *args, axis=None, **kwargs):
        return plot_dos(
            dos_energies=self._phonopy_dict["total_dos_dict"]["frequency_points"],
            dos_total=self._phonopy_dict["total_dos_dict"]["total_dos"],
            *args,
            axis=axis,
            **kwargs,
        )
