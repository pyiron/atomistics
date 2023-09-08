from typing import Optional
import posixpath

import numpy as np
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS
import structuretoolkit

from atomistics.workflows.shared.workflow import Workflow
from atomistics.workflows.phonons.helper import (
    get_supercell_matrix,
    get_hesse_matrix,
    get_band_structure,
    plot_band_structure,
    plot_dos,
)
from atomistics.workflows.phonons.units import VaspToTHz, kJ_mol_to_eV


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
        structure,
        interaction_range=10,
        factor=VaspToTHz,
        displacement=0.01,
        dos_mesh=20,
        primitive_matrix=None,
        number_of_snapshots=None,
    ):
        self._interaction_range = interaction_range
        self._displacement = displacement
        self._dos_mesh = dos_mesh
        self._number_of_snapshots = number_of_snapshots
        self.structure = structure
        self._phonopy_unit_cell = structuretoolkit.common.atoms_to_phonopy(
            self.structure
        )
        self.phonopy = Phonopy(
            unitcell=self._phonopy_unit_cell,
            supercell_matrix=get_supercell_matrix(
                interaction_range=self._interaction_range,
                cell=self._phonopy_unit_cell.get_cell(),
            ),
            primitive_matrix=primitive_matrix,
            factor=factor,
        )
        self.phonopy.generate_displacements(
            distance=self._displacement,
            number_of_snapshots=self._number_of_snapshots,
        )
        self._dos_dict = {}
        self._mesh_dict = {}

    def generate_structures(self):
        return {
            "calc_forces": {
                ind: self._restore_magmoms(structuretoolkit.common.phonopy_to_atoms(sc))
                for ind, sc in enumerate(self.phonopy.supercells_with_displacements)
            }
        }

    def _restore_magmoms(self, structure):
        """
        Args:
            structure (pyiron.atomistics.structure.atoms): input structure

        Returns:
            structure (pyiron_atomistics.atomistics.structure.atoms): output structure with magnetic moments
        """
        if self.structure.has("initial_magmoms"):
            magmoms = self.structure.get_initial_magnetic_moments()
            magmoms = np.tile(
                magmoms,
                np.prod(
                    np.diagonal(
                        get_supercell_matrix(
                            interaction_range=self._interaction_range,
                            cell=self._phonopy_unit_cell.get_cell(),
                        )
                    )
                ).astype(int),
            )
            structure.set_initial_magnetic_moments(magmoms)
        return structure

    def analyse_structures(self, output_dict):
        """

        Returns:

        """
        if "forces" in output_dict.keys():
            output_dict = output_dict["forces"]
        forces_lst = [output_dict[k] for k in sorted(output_dict.keys())]
        self.phonopy.forces = forces_lst
        self.phonopy.produce_force_constants(
            fc_calculator=None if self._number_of_snapshots is None else "alm"
        )
        self.phonopy.run_mesh(mesh=[self._dos_mesh] * 3)
        self._mesh_dict = self.phonopy.get_mesh_dict()
        self.phonopy.run_total_dos()
        self._dos_dict = self.phonopy.get_total_dos_dict()
        return self._mesh_dict, self._dos_dict

    def get_thermal_properties(self, t_min=1, t_max=1500, t_step=50, temperatures=None):
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
        self.phonopy.run_thermal_properties(
            t_step=t_step, t_max=t_max, t_min=t_min, temperatures=temperatures
        )
        tp_dict = self.phonopy.get_thermal_properties_dict()
        tp_dict["free_energy"] *= kJ_mol_to_eV
        return tp_dict

    def get_dynamical_matrix(self):
        """

        Returns:

        """
        return np.real_if_close(self.phonopy.dynamical_matrix.dynamical_matrix)

    def dynamical_matrix_at_q(self, q):
        """

        Args:
            q:

        Returns:

        """
        return np.real_if_close(self.phonopy.get_dynamical_matrix_at_q(q))

    def write_phonopy_force_constants(self, file_name="FORCE_CONSTANTS", cwd=None):
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

    def get_hesse_matrix(self):
        return get_hesse_matrix(force_constants=self.phonopy.force_constants)

    def get_band_structure(
        self, npoints=101, with_eigenvectors=False, with_group_velocities=False
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
            **kwargs
        )

    def plot_dos(self, *args, axis=None, **kwargs):
        return plot_dos(
            dos_energies=self._dos_dict["frequency_points"],
            dos_total=self._dos_dict["total_dos"],
            *args,
            axis=axis,
            **kwargs
        )
