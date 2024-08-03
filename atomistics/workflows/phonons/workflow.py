import posixpath
from typing import Optional

import numpy as np
from ase.atoms import Atoms
from phonopy.file_IO import write_FORCE_CONSTANTS

from atomistics.shared.output import OutputPhonons, OutputThermodynamic
from atomistics.workflows.interface import Workflow
from atomistics.workflows.phonons.helper import (
    analyse_structures_helper,
    generate_structures_helper,
    get_band_structure,
    get_hesse_matrix,
    get_thermal_properties,
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
        primitive_matrix: Optional[np.ndarray] = None,
        number_of_snapshots: Optional[int] = None,
    ):
        """
        Initialize the PhonopyWorkflow.

        Args:
            structure (Atoms): The structure used in the reference job.
            interaction_range (float, optional): The interaction range. Defaults to 10.0.
            factor (float, optional): The conversion factor. Defaults to VaspToTHz.
            displacement (float, optional): The displacement. Defaults to 0.01.
            dos_mesh (int, optional): The DOS mesh. Defaults to 20.
            primitive_matrix (np.ndarray, optional): The primitive matrix. Defaults to None.
            number_of_snapshots (int, optional): The number of snapshots. Defaults to None.
        """
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
        """
        Generate structures.

        Returns:
            dict: The generated structures.
        """
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
        Analyse structures.

        Args:
            output_dict (dict): The output dictionary.
            output_keys (tuple[str], optional): The output keys. Defaults to OutputPhonons.keys().

        Returns:
            dict: The analysed structures.
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
        temperatures: Optional[np.ndarray] = None,
        cutoff_frequency: Optional[float] = None,
        pretend_real: bool = False,
        band_indices: Optional[np.ndarray] = None,
        is_projection: bool = False,
        output_keys: tuple[str] = OutputThermodynamic.keys(),
    ) -> dict:
        """
        Get thermal properties.

        Args:
            t_min (float, optional): The minimum sample temperature. Defaults to 1.0.
            t_max (float, optional): The maximum sample temperature. Defaults to 1500.0.
            t_step (float, optional): The temperature sample interval. Defaults to 50.0.
            temperatures (np.ndarray, optional): Custom array of temperature samples. Defaults to None.
            cutoff_frequency (float, optional): The cutoff frequency. Defaults to None.
            pretend_real (bool, optional): Whether to pretend real. Defaults to False.
            band_indices (np.ndarray, optional): The band indices. Defaults to None.
            is_projection (bool, optional): Whether it is a projection. Defaults to False.
            output_keys (tuple[str], optional): The output keys. Defaults to OutputThermodynamic.keys().

        Returns:
            dict: The thermal properties.
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
        Get the dynamical matrix.

        Args:
            npoints (int, optional): The number of points. Defaults to 101.

        Returns:
            np.ndarray: The dynamical matrix.
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
        Get the dynamical matrix at a given q.

        Args:
            q (np.ndarray): The q value.

        Returns:
            np.ndarray: The dynamical matrix.
        """
        return np.real_if_close(self.phonopy.get_dynamical_matrix_at_q(q))

    def write_phonopy_force_constants(
        self, file_name: str = "FORCE_CONSTANTS", cwd: Optional[str] = None
    ):
        """
        Write the Phonopy force constants.

        Args:
            file_name (str, optional): The file name. Defaults to "FORCE_CONSTANTS".
            cwd (str, optional): The current working directory. Defaults to None.
        """
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)
        write_FORCE_CONSTANTS(
            force_constants=self.phonopy.force_constants, filename=file_name
        )

    def get_hesse_matrix(self) -> np.ndarray:
        """
        Get the Hesse matrix.

        Returns:
            np.ndarray: The Hesse matrix.
        """
        return get_hesse_matrix(force_constants=self.phonopy.force_constants)

    def get_band_structure(
        self,
        npoints: int = 101,
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False,
    ):
        """
        Get the band structure.

        Args:
            npoints (int, optional): The number of points. Defaults to 101.
            with_eigenvectors (bool, optional): Whether to include eigenvectors. Defaults to False.
            with_group_velocities (bool, optional): Whether to include group velocities. Defaults to False.

        Returns:
            [type]: [description]
        """
        return get_band_structure(
            phonopy=self.phonopy,
            npoints=npoints,
            with_eigenvectors=with_eigenvectors,
            with_group_velocities=with_group_velocities,
        )

    def plot_band_structure(
        self, axis=None, *args, label: Optional[str] = None, **kwargs
    ):
        """
        Plot the band structure.

        Args:
            axis ([type], optional): The axis. Defaults to None.
            label (str, optional): The label. Defaults to None.

        Returns:
            [type]: [description]
        """
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
        """
        Plot the DOS.

        Args:
            axis ([type], optional): The axis. Defaults to None.

        Returns:
            [type]: [description]
        """
        return plot_dos(
            dos_energies=self._phonopy_dict["total_dos_dict"]["frequency_points"],
            dos_total=self._phonopy_dict["total_dos_dict"]["total_dos"],
            *args,
            axis=axis,
            **kwargs,
        )
