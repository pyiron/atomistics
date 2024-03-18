from ase.atoms import Atoms
import numpy as np

from atomistics.shared.output import OutputThermodynamic
from atomistics.workflows.evcurve.workflow import (
    EnergyVolumeCurveWorkflow,
    fit_ev_curve,
    _strain_axes,
)
from atomistics.workflows.phonons.workflow import PhonopyWorkflow
from atomistics.workflows.phonons.helper import get_supercell_matrix
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
    volume_lst: np.ndarray,
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
        strain: phono.get_thermal_properties(
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
            for freqs, w in zip(
                phono.phonopy.mesh.frequencies, phono.phonopy.mesh.weights
            ):
                freqs = np.array(freqs) * THzToEv
                cond = freqs > cutoff_frequency
                t_property += (
                    np.sum(
                        get_free_energy_classical(frequency=freqs[cond], temperature=t)
                    )
                    * w
                )
            t_property_lst.append(
                t_property / np.sum(phono.phonopy.mesh.weights) * EvTokJmol
            )
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
        # Phonopy internally repeats structures that are "too small"
        # Here we manually guarantee that all structures passed are big enough
        # This provides some computational efficiency for classical calculations
        # And for quantum calculations _ensures_ that force matrices and energy/atom
        # get treated with the same kmesh
        self._repeat_vector = np.array(
            np.diag(
                get_supercell_matrix(
                    interaction_range=interaction_range,
                    cell=structure.cell.array,
                )
            ),
            dtype=int,
        )

    def generate_structures(self) -> dict:
        task_dict = {"calc_forces": {}}
        for strain in self._get_strains():
            strain_ind = 1 + np.round(strain, 7)
            basis = _strain_axes(
                structure=self.structure, axes=self.axes, volume_strain=strain
            )
            structure_ev = basis.repeat(self._repeat_vector)
            self._structure_dict[strain_ind] = structure_ev
            self._phonopy_dict[strain_ind] = PhonopyWorkflow(
                structure=basis,
                interaction_range=self._interaction_range,
                factor=self._factor,
                displacement=self._displacement,
                dos_mesh=self._dos_mesh,
                primitive_matrix=self._primitive_matrix,
                number_of_snapshots=self._number_of_snapshots,
            )
            structure_task_dict = self._phonopy_dict[strain_ind].generate_structures()
            task_dict["calc_forces"].update(
                {
                    (strain_ind, key): structure_phono
                    for key, structure_phono in structure_task_dict[
                        "calc_forces"
                    ].items()
                }
            )
        task_dict["calc_energy"] = self._structure_dict
        return task_dict

    def analyse_structures(
        self,
        output_dict: dict,
        output_keys: tuple[str] = ("force_constants", "mesh_dict"),
    ):
        self._eng_internal_dict = output_dict["energy"]
        phonopy_collect_dict = {
            strain: phono.analyse_structures(
                output_dict={
                    k: v for k, v in output_dict["forces"].items() if strain in k
                },
                output_keys=output_keys,
            )
            for strain, phono in self._phonopy_dict.items()
        }
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
            volume_lst=np.array(self.get_volume_lst()) / np.prod(self._repeat_vector),
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
