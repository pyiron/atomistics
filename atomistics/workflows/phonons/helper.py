from typing import Optional

from ase.atoms import Atoms
import numpy as np
from phonopy import Phonopy
import scipy.constants
import structuretoolkit

from atomistics.shared.output import OutputThermodynamic, OutputPhonons
from atomistics.workflows.phonons.units import VaspToTHz, kJ_mol_to_eV


class PhonopyProperties(object):
    def __init__(
        self,
        phonopy_instance: Phonopy,
        dos_mesh: np.ndarray,
        shift=None,
        is_time_reversal: bool = True,
        is_mesh_symmetry: bool = True,
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False,
        is_gamma_center: bool = False,
        number_of_snapshots: int = None,
        sigma: float = None,
        freq_min: float = None,
        freq_max: float = None,
        freq_pitch: float = None,
        use_tetrahedron_method: bool = True,
        npoints: int = 101,
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

    def mesh_dict(self) -> dict:
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

    def band_structure_dict(self) -> dict:
        if self._band_structure_dict is None:
            self._calc_band_structure()
        return self._band_structure_dict

    def total_dos_dict(self) -> dict:
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

    def dynamical_matrix(self) -> np.ndarray:
        if self._band_structure_dict is None:
            self._calc_band_structure()
        return self._phonopy.dynamical_matrix.dynamical_matrix

    def force_constants(self) -> np.ndarray:
        if self._force_constants is None:
            self._calc_force_constants()
        return self._force_constants


class PhonopyThermalProperties(object):
    def __init__(self, phonopy_instance: Phonopy):
        self._phonopy = phonopy_instance
        self._thermal_properties = phonopy_instance.get_thermal_properties_dict()

    def free_energy(self) -> np.ndarray:
        return self._thermal_properties["free_energy"] * kJ_mol_to_eV

    def temperatures(self) -> np.ndarray:
        return self._thermal_properties["temperatures"]

    def entropy(self) -> np.ndarray:
        return self._thermal_properties["entropy"]

    def heat_capacity(self) -> np.ndarray:
        return self._thermal_properties["heat_capacity"]

    def volumes(self) -> np.ndarray:
        return np.array(
            [self._phonopy.unitcell.get_volume()]
            * len(self._thermal_properties["temperatures"])
        )


def restore_magmoms(
    structure_with_magmoms: Atoms,
    structure: Atoms,
    interaction_range: float,
    cell: np.ndarray,
) -> Atoms:
    """
    Args:
        structure_with_magmoms (ase.atoms.Atoms): input structure with magnetic moments
        structure (ase.atoms.Atoms): input structure without magnetic moments
        interaction_range (float):
        cell (np.ndarray):

    Returns:
        structure (ase.atoms.Atoms): output structure with magnetic moments
    """
    if structure_with_magmoms.has("initial_magmoms"):
        magmoms = structure_with_magmoms.get_initial_magnetic_moments()
        magmoms = np.tile(
            magmoms,
            np.prod(
                np.diagonal(
                    get_supercell_matrix(
                        interaction_range=interaction_range,
                        cell=cell,
                    )
                )
            ).astype(int),
        )
        structure.set_initial_magnetic_moments(magmoms)
    return structure


def generate_structures_helper(
    structure: Atoms,
    primitive_matrix: np.ndarray = None,
    displacement: float = 0.01,
    number_of_snapshots: int = None,
    interaction_range: float = 10.0,
    factor: float = VaspToTHz,
):
    unitcell = structuretoolkit.common.atoms_to_phonopy(structure)
    phonopy_obj = Phonopy(
        unitcell=unitcell,
        supercell_matrix=get_supercell_matrix(
            interaction_range=interaction_range,
            cell=unitcell.get_cell(),
        ),
        primitive_matrix=primitive_matrix,
        factor=factor,
    )
    phonopy_obj.generate_displacements(
        distance=displacement,
        number_of_snapshots=number_of_snapshots,
    )
    structure_dict = {
        ind: restore_magmoms(
            structure_with_magmoms=structure,
            structure=structuretoolkit.common.phonopy_to_atoms(sc),
            interaction_range=interaction_range,
            cell=unitcell.get_cell(),
        )
        for ind, sc in enumerate(phonopy_obj.supercells_with_displacements)
    }
    return phonopy_obj, structure_dict


def analyse_structures_helper(
    phonopy: Phonopy,
    output_dict: dict,
    dos_mesh: int = 20,
    number_of_snapshots: int = None,
    output_keys: tuple[str] = OutputPhonons.keys(),
) -> dict:
    """

    Returns:

    """
    if "forces" in output_dict.keys():
        output_dict = output_dict["forces"]
    forces_lst = [output_dict[k] for k in sorted(output_dict.keys())]
    phonopy.forces = forces_lst
    phono = PhonopyProperties(
        phonopy_instance=phonopy,
        dos_mesh=dos_mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        with_eigenvectors=False,
        with_group_velocities=False,
        is_gamma_center=False,
        number_of_snapshots=number_of_snapshots,
        sigma=None,
        freq_min=None,
        freq_max=None,
        freq_pitch=None,
        use_tetrahedron_method=True,
        npoints=101,
    )
    return OutputPhonons(**{k: getattr(phono, k) for k in OutputPhonons.keys()}).get(
        output_keys=output_keys
    )


def get_thermal_properties(
    phonopy: Phonopy,
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
    phonopy.run_thermal_properties(
        t_step=t_step,
        t_max=t_max,
        t_min=t_min,
        temperatures=temperatures,
        cutoff_frequency=cutoff_frequency,
        pretend_real=pretend_real,
        band_indices=band_indices,
        is_projection=is_projection,
    )
    phono = PhonopyThermalProperties(phonopy_instance=phonopy)
    return OutputThermodynamic(
        **{k: getattr(phono, k) for k in OutputThermodynamic.keys()}
    ).get(output_keys=output_keys)


def get_supercell_matrix(interaction_range: float, cell: np.ndarray) -> np.ndarray:
    supercell_range = np.ceil(
        interaction_range / np.array([np.linalg.norm(vec) for vec in cell])
    )
    return np.eye(3) * supercell_range


def get_hesse_matrix(force_constants: np.ndarray) -> np.ndarray:
    """

    Returns:

    """
    unit_conversion = (
        scipy.constants.physical_constants["Hartree energy in eV"][0]
        / scipy.constants.physical_constants["Bohr radius"][0] ** 2
        * scipy.constants.angstrom**2
    )
    force_shape = np.shape(force_constants)
    force_reshape = force_shape[0] * force_shape[2]
    return (
        np.transpose(force_constants, (0, 2, 1, 3)).reshape(
            (force_reshape, force_reshape)
        )
        / unit_conversion
    )


def plot_dos(
    dos_energies: np.ndarray, dos_total: np.ndarray, *args, axis=None, **kwargs
):
    """
    Plot the DOS.

    If "label" is present in `kwargs` a legend is added to the plot automatically.

    Args:
        axis (optional): matplotlib axis to use, if None create a new one
        *args: passed to `axis.plot`
        **kwargs: passed to `axis.plot`

    Returns:
        matplotlib.axes._subplots.AxesSubplot: axis with the plot
    """
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots(1, 1)
    axis.plot(dos_energies, dos_total, *args, **kwargs)
    axis.set_xlabel("Frequency [THz]")
    axis.set_ylabel("DOS")
    axis.set_title("Phonon DOS vs Energy")
    if "label" in kwargs:
        axis.legend()
    return axis


def get_band_structure(
    phonopy: Phonopy,
    npoints: int = 101,
    with_eigenvectors: bool = False,
    with_group_velocities: bool = False,
):
    """
    Calculate band structure with automatic path through reciprocal space.

    Can only be called after job is finished.

    Args:
        npoints (int, optional):  Number of sample points between high symmetry points.
        with_eigenvectors (boolean, optional):  Calculate eigenvectors, too
        with_group_velocities (boolean, optional):  Calculate group velocities, too

    Returns:
        :class:`dict` of the results from phonopy under the following keys
            - 'qpoints':  list of (npoints, 3), samples paths in reciprocal space
            - 'distances':  list of (npoints,), distance along the paths in reciprocal space
            - 'frequencies':  list of (npoints, band), phonon frequencies
            - 'eigenvectors':  list of (npoints, band, band//3, 3), phonon eigenvectors
            - 'group_velocities': list of (npoints, band), group velocities
        where band is the number of bands (number of atoms * 3).  Each entry is a list of arrays, and each array
        corresponds to one path between two high-symmetry points automatically picked by Phonopy and may be of
        different length than other paths.  As compared to the phonopy output this method also reshapes the
        eigenvectors so that they directly have the same shape as the underlying structure.

    Raises:
        :exception:`ValueError`: method is called on a job that is not finished or aborted
    """
    phonopy.auto_band_structure(
        npoints,
        with_eigenvectors=with_eigenvectors,
        with_group_velocities=with_group_velocities,
    )
    results = phonopy.get_band_structure_dict()
    if results["eigenvectors"] is not None:
        # see https://phonopy.github.io/phonopy/phonopy-module.html#eigenvectors for the way phonopy stores the
        # eigenvectors
        results["eigenvectors"] = [
            e.transpose(0, 2, 1).reshape(*e.shape[:2], -1, 3)
            for e in results["eigenvectors"]
        ]
    return results


def plot_band_structure(
    results: dict,
    path_connections: list[str],
    labels: str,
    axis=None,
    *args,
    label: Optional[str] = None,
    **kwargs,
):
    """
    Plot bandstructure calculated with :meth:`.get_bandstructure`.

    If :meth:`.get_bandstructure` hasn't been called before, it is automatically called with the default arguments.

    If `label` is passed a legend is added automatically.

    Args:
        axis (matplotlib axis, optional): plot to this axis, if not given a new one is created.
        *args: passed through to matplotlib.pyplot.plot when plotting the dispersion
        label (str, optional): label for dispersion line
        **kwargs: passed through to matplotlib.pyplot.plot when plotting the dispersion

    Returns:
        matplib axis: the axis the figure has been drawn to, if axis is given the same object is returned
    """
    # label is it's own argument because if you try to pass it via **kwargs every line would get the label giving a
    # messy legend
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots(1, 1)

    distances = results["distances"]
    frequencies = results["frequencies"]

    if "color" not in kwargs:
        kwargs["color"] = "black"

    offset = 0
    tick_positions = [distances[0][0]]
    for di, fi, ci in zip(distances, frequencies, path_connections):
        axis.axvline(tick_positions[-1], color="black", linestyle="dotted", alpha=0.5)
        line, *_ = axis.plot(offset + di, fi, *args, **kwargs)
        tick_positions.append(di[-1] + offset)
        if not ci:
            offset += 0.05
            axis.axvline(
                tick_positions[-1], color="black", linestyle="dotted", alpha=0.5
            )
            tick_positions.append(di[-1] + offset)
    if label is not None:
        line.set_label(label)
        axis.legend()
    axis.set_xticks(tick_positions[:-1])
    axis.set_xticklabels(labels)
    axis.set_xlabel("Bandpath")
    axis.set_ylabel("Frequency [THz]")
    axis.set_title("Bandstructure")
    return axis
