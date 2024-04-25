from typing import Optional, List

from ase.atoms import Atoms
import numpy as np

from atomistics.shared.output import OutputThermodynamic
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow
from atomistics.workflows.evcurve.helper import (
    get_strains,
    get_volume_lst,
    fit_ev_curve,
    _strain_axes,
)
from atomistics.workflows.phonons.helper import (
    get_supercell_matrix,
    analyse_structures_helper as analyse_structures_phonopy_helper,
    get_thermal_properties as get_thermal_properties_phonopy,
    generate_structures_helper as generate_structures_phonopy_helper,
)
from atomistics.workflows.phonons.units import (
    VaspToTHz,
    kJ_mol_to_eV,
    THzToEv,
    kb,
    EvTokJmol,
)


def get_free_energy_classical(
    frequency: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    return kb * temperature * np.log(frequency / (kb * temperature))


def get_thermal_properties(
    eng_internal_dict: dict,
    phonopy_dict: dict,
    structure_dict: dict,
    repeat_vector: np.ndarray,
    fit_type: str,
    fit_order: int,
    t_min: float = 1.0,
    t_max: float = 1500.0,
    t_step: float = 50.0,
    temperatures: np.ndarray = None,
    cutoff_frequency: float = None,
    pretend_real: bool = False,
    band_indices: np.ndarray = None,
    is_projection: bool = False,
    quantum_mechanical: bool = True,
    output_keys: tuple[str] = OutputThermodynamic.keys(),
) -> dict:
    """
    Returns thermal properties at constant volume in the given temperature range.  Can only be called after job
    successfully ran.

    Args:
        t_min (float): minimum sample temperature
        t_max (float): maximum sample temperature
        t_step (int):  temperature sample interval
        temperatures (array_like, float):  custom array of temperature samples, if given t_min, t_max, t_step are
                                           ignored.

    Returns:
        :class:`Thermal`: thermal properties as returned by Phonopy
    """
    volume_lst = np.array(get_volume_lst(structure_dict=structure_dict)) / np.prod(
        repeat_vector
    )
    eng_internal_dict = {
        key: value / np.prod(repeat_vector) for key, value in eng_internal_dict.items()
    }
    if quantum_mechanical:
        tp_collect_dict = _get_thermal_properties_quantum_mechanical(
            phonopy_dict=phonopy_dict,
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
    else:
        if is_projection:
            raise ValueError(
                "is_projection!=False is incompatible to quantum_mechanical=False."
            )
        if pretend_real:
            raise ValueError(
                "pretend_real!=False is incompatible to quantum_mechanical=False."
            )
        if band_indices is not None:
            raise ValueError(
                "band_indices!=None is incompatible to quantum_mechanical=False."
            )
        tp_collect_dict = _get_thermal_properties_classical(
            phonopy_dict=phonopy_dict,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            cutoff_frequency=cutoff_frequency,
        )

    temperatures = tp_collect_dict[1.0]["temperatures"]
    strain_lst = eng_internal_dict.keys()
    eng_int_lst = np.array(list(eng_internal_dict.values()))

    vol_lst, eng_lst = [], []
    for i, temp in enumerate(temperatures):
        free_eng_lst = (
            np.array([tp_collect_dict[s]["free_energy"][i] for s in strain_lst])
            + eng_int_lst
        )
        fit_dict = fit_ev_curve(
            volume_lst=volume_lst,
            energy_lst=free_eng_lst,
            fit_type=fit_type,
            fit_order=fit_order,
        )
        eng_lst.append(fit_dict["energy_eq"])
        vol_lst.append(fit_dict["volume_eq"])

    if (
        not quantum_mechanical
    ):  # heat capacity and entropy are not yet implemented for the classical approach.
        output_keys = ["free_energy", "temperatures", "volumes"]
    qhp = QuasiHarmonicThermalProperties(
        temperatures=temperatures,
        thermal_properties_dict=tp_collect_dict,
        strain_lst=strain_lst,
        volumes_lst=volume_lst,
        volumes_selected_lst=vol_lst,
    )
    return OutputThermodynamic(
        **{k: getattr(qhp, k) for k in OutputThermodynamic.keys()}
    ).get(output_keys=output_keys)


def _get_thermal_properties_quantum_mechanical(
    phonopy_dict: dict,
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
        t_step (int):  temperature sample interval
        temperatures (array_like, float):  custom array of temperature samples, if given t_min, t_max, t_step are
                                           ignored.

    Returns:
        :class:`Thermal`: thermal properties as returned by Phonopy
    """
    return {
        strain: get_thermal_properties_phonopy(
            phonopy=phono,
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            temperatures=temperatures,
            cutoff_frequency=cutoff_frequency,
            pretend_real=pretend_real,
            band_indices=band_indices,
            is_projection=is_projection,
            output_keys=output_keys,
        )
        for strain, phono in phonopy_dict.items()
    }


def _get_thermal_properties_classical(
    phonopy_dict: dict,
    t_min: float = 1.0,
    t_max: float = 1500.0,
    t_step: float = 50.0,
    temperatures: np.ndarray = None,
    cutoff_frequency: float = None,
) -> dict:
    """
    Returns thermal properties at constant volume in the given temperature range.  Can only be called after job
    successfully ran.

    Args:
        t_min (float): minimum sample temperature
        t_max (float): maximum sample temperature
        t_step (int):  temperature sample interval
        temperatures (array_like, float):  custom array of temperature samples, if given t_min, t_max, t_step are
                                           ignored.

    Returns:
        :class:`Thermal`: thermal properties as returned by Phonopy
    """
    if temperatures is None:
        temperatures = np.arange(t_min, t_max, t_step)
    if cutoff_frequency is None or cutoff_frequency < 0:
        cutoff_frequency = 0.0
    else:
        cutoff_frequency = cutoff_frequency * THzToEv
    tp_collect_dict = {}
    for strain, phono in phonopy_dict.items():
        t_property_lst = []
        for t in temperatures:
            t_property = 0.0
            for freqs, w in zip(phono.mesh.frequencies, phono.mesh.weights):
                freqs = np.array(freqs) * THzToEv
                cond = freqs > cutoff_frequency
                t_property += (
                    np.sum(
                        get_free_energy_classical(frequency=freqs[cond], temperature=t)
                    )
                    * w
                )
            t_property_lst.append(t_property / np.sum(phono.mesh.weights) * EvTokJmol)
        tp_collect_dict[strain] = {
            "temperatures": temperatures,
            "free_energy": np.array(t_property_lst) * kJ_mol_to_eV,
        }
    return tp_collect_dict


class QuasiHarmonicThermalProperties(object):
    def __init__(
        self,
        temperatures: np.ndarray,
        thermal_properties_dict: dict,
        strain_lst: np.ndarray,
        volumes_lst: np.ndarray,
        volumes_selected_lst: np.ndarray,
    ):
        self._temperatures = temperatures
        self._thermal_properties_dict = thermal_properties_dict
        self._strain_lst = strain_lst
        self._volumes_lst = volumes_lst
        self._volumes_selected_lst = volumes_selected_lst

    def get_property(self, thermal_property: str) -> np.ndarray:
        return np.array(
            [
                np.poly1d(np.polyfit(self._volumes_lst, q_over_v, 1))(vol_opt)
                for q_over_v, vol_opt in zip(
                    np.array(
                        [
                            self._thermal_properties_dict[s][thermal_property]
                            for s in self._strain_lst
                        ]
                    ).T,
                    self._volumes_selected_lst,
                )
            ]
        )

    def free_energy(self) -> np.ndarray:
        return self.get_property(thermal_property="free_energy")

    def temperatures(self) -> np.ndarray:
        return self._temperatures

    def entropy(self) -> np.ndarray:
        return self.get_property(thermal_property="entropy")

    def heat_capacity(self) -> np.ndarray:
        return self.get_property(thermal_property="heat_capacity")

    def volumes(self) -> np.ndarray:
        return self._volumes_selected_lst


def generate_structures_helper(
    structure: Atoms,
    vol_range: Optional[float] = None,
    num_points: Optional[int] = None,
    strain_lst: Optional[List[float]] = None,
    displacement: float = 0.01,
    number_of_snapshots: int = None,
    interaction_range: float = 10.0,
    factor: float = VaspToTHz,
):
    # Phonopy internally repeats structures that are "too small"
    # Here we manually guarantee that all structures passed are big enough
    # This provides some computational efficiency for classical calculations
    # And for quantum calculations _ensures_ that force matrices and energy/atom
    # get treated with the same kmesh
    repeat_vector = np.array(
        np.diag(
            get_supercell_matrix(
                interaction_range=interaction_range,
                cell=structure.cell.array,
            )
        ),
        dtype=int,
    )
    structure_energy_dict, structure_forces_dict, phonopy_dict = {}, {}, {}
    for strain in get_strains(
        vol_range=vol_range,
        num_points=num_points,
        strain_lst=strain_lst,
    ):
        strain_ind = 1 + np.round(strain, 7)
        basis = _strain_axes(
            structure=structure, axes=("x", "y", "z"), volume_strain=strain
        )
        structure_energy_dict[strain_ind] = basis.repeat(repeat_vector)
        phonopy_obj, structure_task_dict = generate_structures_phonopy_helper(
            structure=basis,
            displacement=displacement,
            number_of_snapshots=number_of_snapshots,
            interaction_range=interaction_range,
            factor=factor,
        )
        phonopy_dict[strain_ind] = phonopy_obj
        structure_forces_dict.update(
            {
                (strain_ind, key): structure_phono
                for key, structure_phono in structure_task_dict.items()
            }
        )
    return phonopy_dict, repeat_vector, structure_energy_dict, structure_forces_dict


def analyse_structures_helper(
    phonopy_dict: dict,
    output_dict: dict,
    dos_mesh: int = 20,
    number_of_snapshots: int = None,
    output_keys: tuple[str] = ("force_constants", "mesh_dict"),
):
    eng_internal_dict = output_dict["energy"]
    phonopy_collect_dict = {
        strain: analyse_structures_phonopy_helper(
            phonopy=phono,
            output_dict={k: v for k, v in output_dict["forces"].items() if strain in k},
            dos_mesh=dos_mesh,
            number_of_snapshots=number_of_snapshots,
            output_keys=output_keys,
        )
        for strain, phono in phonopy_dict.items()
    }
    return eng_internal_dict, phonopy_collect_dict


class QuasiHarmonicWorkflow(EnergyVolumeCurveWorkflow):
    def __init__(
        self,
        structure: Atoms,
        num_points: int = 11,
        vol_range: float = 0.05,
        fit_type: str = "polynomial",
        fit_order: int = 3,
        interaction_range: float = 10.0,
        factor: float = VaspToTHz,
        displacement: float = 0.01,
        dos_mesh: int = 20,
        primitive_matrix: np.ndarray = None,
        number_of_snapshots: int = None,
    ):
        super().__init__(
            structure=structure,
            num_points=num_points,
            fit_type=fit_type,
            fit_order=fit_order,
            vol_range=vol_range,
            axes=["x", "y", "z"],
            strains=None,
        )
        self._interaction_range = interaction_range
        self._displacement = displacement
        self._dos_mesh = dos_mesh
        self._number_of_snapshots = number_of_snapshots
        self._factor = factor
        self._primitive_matrix = primitive_matrix
        self._phonopy_dict = {}
        self._eng_internal_dict = None
        self._repeat_vector = None

    def generate_structures(self) -> dict:
        (
            self._phonopy_dict,
            self._repeat_vector,
            structure_energy_dict,
            structure_forces_dict,
        ) = generate_structures_helper(
            structure=self.structure,
            vol_range=self.vol_range,
            num_points=self.num_points,
            strain_lst=self.strains,
            displacement=self._displacement,
            number_of_snapshots=self._number_of_snapshots,
            interaction_range=self._interaction_range,
            factor=self._factor,
        )
        self._structure_dict = structure_energy_dict
        return {
            "calc_energy": structure_energy_dict,
            "calc_forces": structure_forces_dict,
        }

    def analyse_structures(
        self,
        output_dict: dict,
        output_keys: tuple[str] = ("force_constants", "mesh_dict"),
    ):
        self._eng_internal_dict, phonopy_collect_dict = analyse_structures_helper(
            phonopy_dict=self._phonopy_dict,
            output_dict=output_dict,
            dos_mesh=self._dos_mesh,
            number_of_snapshots=self._number_of_snapshots,
            output_keys=output_keys,
        )
        return self._eng_internal_dict, phonopy_collect_dict

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
        quantum_mechanical: bool = True,
        output_keys: tuple[str] = OutputThermodynamic.keys(),
    ):
        """
        Returns thermal properties at constant volume in the given temperature range.  Can only be called after job
        successfully ran.

        Args:
            t_min (float): minimum sample temperature
            t_max (float): maximum sample temperature
            t_step (int):  temperature sample interval
            temperatures (array_like, float):  custom array of temperature samples, if given t_min, t_max, t_step are
                                               ignored.

        Returns:
            :class:`Thermal`: thermal properties as returned by Phonopy
        """
        if self._eng_internal_dict is None:
            raise ValueError(
                "Please first execute analyse_output() before calling get_thermal_properties()."
            )
        return get_thermal_properties(
            eng_internal_dict=self._eng_internal_dict,
            phonopy_dict=self._phonopy_dict,
            structure_dict=self._structure_dict,
            repeat_vector=self._repeat_vector,
            fit_type=self.fit_type,
            fit_order=self.fit_order,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            cutoff_frequency=cutoff_frequency,
            pretend_real=pretend_real,
            band_indices=band_indices,
            is_projection=is_projection,
            quantum_mechanical=quantum_mechanical,
            output_keys=OutputThermodynamic.keys(),
        )
