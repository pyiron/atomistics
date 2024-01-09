from typing import Optional
import posixpath

import numpy as np
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS
import structuretoolkit

from atomistics.shared.output import OutputThermodynamic, OutputPhonons
from atomistics.workflows.interface import Workflow
from atomistics.workflows.phonons.helper import (
    get_supercell_matrix,
    get_hesse_matrix,
    get_band_structure,
    plot_band_structure,
    plot_dos,
)
from atomistics.workflows.phonons.units import VaspToTHz, kJ_mol_to_eV


class PhonopyProperties(object):
    def __init__(
        self,
        phonopy_instance,
        dos_mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        with_eigenvectors=False,
        with_group_velocities=False,
        is_gamma_center=False,
        number_of_snapshots=None,
        sigma=None,
        freq_min=None,
        freq_max=None,
        freq_pitch=None,
        use_tetrahedron_method=True,
        npoints=101,
    ):
        self._phonopy = phonopy_instance
        self._sigma = sigma
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._freq_pitch = freq_pitch
        self._use_tetrahedron_method = use_tetrahedron_method
        self._npoints = npoints
        self._with_eigenvectors = with_eigenvectors
        self._with_group_velocities = with_group_velocities
        self._dos_mesh = dos_mesh
        self._shift = shift
        self._is_time_reversal = is_time_reversal
        self._is_mesh_symmetry = is_mesh_symmetry
        self._with_eigenvectors = with_eigenvectors
        self._with_group_velocities = with_group_velocities
        self._is_gamma_center = is_gamma_center
        self._number_of_snapshots = number_of_snapshots
        self._total_dos = None
        self._band_structure_dict = None
        self._mesh_dict = None
        self._force_constants = None

    def _calc_band_structure(self):
        self._phonopy.auto_band_structure(
            npoints=self._npoints,
            with_eigenvectors=self._with_eigenvectors,
            with_group_velocities=self._with_group_velocities,
        )
        self._band_structure_dict = self._phonopy.get_band_structure_dict()

    def _calc_force_constants(self):
        self._phonopy.produce_force_constants(
            fc_calculator=None if self._number_of_snapshots is None else "alm"
        )
        self._force_constants = self._phonopy.force_constants

    def mesh_dict(self):
        if self._force_constants is None:
            self._calc_force_constants()
        if self._mesh_dict is None:
            self._phonopy.run_mesh(
                mesh=[self._dos_mesh] * 3,
                shift=self._shift,
                is_time_reversal=self._is_time_reversal,
                is_mesh_symmetry=self._is_mesh_symmetry,
                with_eigenvectors=self._with_eigenvectors,
                with_group_velocities=self._with_group_velocities,
                is_gamma_center=self._is_gamma_center,
            )
            self._mesh_dict = self._phonopy.get_mesh_dict()
        return self._mesh_dict

    def band_structure_dict(self):
        if self._band_structure_dict is None:
            self._calc_band_structure()
        return self._band_structure_dict

    def total_dos_dict(self):
        if self._total_dos is None:
            self._phonopy.run_total_dos(
                sigma=self._sigma,
                freq_min=self._freq_min,
                freq_max=self._freq_max,
                freq_pitch=self._freq_pitch,
                use_tetrahedron_method=self._use_tetrahedron_method,
            )
            self._total_dos = self._phonopy.get_total_dos_dict()
        return self._total_dos

    def dynamical_matrix(self):
        if self._band_structure_dict is None:
            self._calc_band_structure()
        return self._phonopy.dynamical_matrix.dynamical_matrix

    def force_constants(self):
        if self._force_constants is None:
            self._calc_force_constants()
        return self._force_constants


class PhonopyThermalProperties(object):
    def __init__(self, phonopy_instance):
        self._phonopy = phonopy_instance
        self._thermal_properties = phonopy_instance.get_thermal_properties_dict()

    def free_energy(self):
        return self._thermal_properties["free_energy"] * kJ_mol_to_eV

    def temperatures(self):
        return self._thermal_properties["temperatures"]

    def entropy(self):
        return self._thermal_properties["entropy"]

    def heat_capacity(self):
        return self._thermal_properties["heat_capacity"]

    def volumes(self):
        return np.array(
            [self._phonopy.unitcell.get_volume()]
            * len(self._thermal_properties["temperatures"])
        )


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
        self._phonopy_dict = {}

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

    def analyse_structures(self, output_dict, output_keys=OutputPhonons.keys()):
        """

        Returns:

        """
        if "forces" in output_dict.keys():
            output_dict = output_dict["forces"]
        forces_lst = [output_dict[k] for k in sorted(output_dict.keys())]
        self.phonopy.forces = forces_lst
        phono = PhonopyProperties(
            phonopy_instance=self.phonopy,
            dos_mesh=self._dos_mesh,
            shift=None,
            is_time_reversal=True,
            is_mesh_symmetry=True,
            with_eigenvectors=False,
            with_group_velocities=False,
            is_gamma_center=False,
            number_of_snapshots=self._number_of_snapshots,
            sigma=None,
            freq_min=None,
            freq_max=None,
            freq_pitch=None,
            use_tetrahedron_method=True,
            npoints=101,
        )
        self._phonopy_dict = OutputPhonons(
            **{k: getattr(phono, k) for k in OutputPhonons.keys()}
        ).get(output_keys=output_keys)
        return self._phonopy_dict

    def get_thermal_properties(
        self,
        t_min=1,
        t_max=1500,
        t_step=50,
        temperatures=None,
        cutoff_frequency=None,
        pretend_real=False,
        band_indices=None,
        is_projection=False,
        output_keys=OutputThermodynamic.keys(),
    ):
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
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            temperatures=temperatures,
            cutoff_frequency=cutoff_frequency,
            pretend_real=pretend_real,
            band_indices=band_indices,
            is_projection=is_projection,
        )
        phono = PhonopyThermalProperties(phonopy_instance=self.phonopy)
        return OutputThermodynamic(
            **{k: getattr(phono, k) for k in OutputThermodynamic.keys()}
        ).get(output_keys=output_keys)

    def get_dynamical_matrix(self, npoints=101):
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
