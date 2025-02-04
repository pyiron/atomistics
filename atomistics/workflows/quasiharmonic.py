from typing import Optional

import numpy as np
from ase.atoms import Atoms

from atomistics.shared.output import OutputThermodynamic
from atomistics.workflows.evcurve.helper import (
    _strain_axes,
    fit_ev_curve,
    get_strains,
    get_volume_lst,
)
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow
from atomistics.workflows.phonons.helper import (
    analyse_structures_helper as analyse_structures_phonopy_helper,
)
from atomistics.workflows.phonons.helper import (
    generate_structures_helper as generate_structures_phonopy_helper,
)
from atomistics.workflows.phonons.helper import (
    get_supercell_matrix,
)
from atomistics.workflows.phonons.helper import (
    get_thermal_properties as get_thermal_properties_phonopy,
)
from atomistics.workflows.phonons.units import (
    EvTokJmol,
    THzToEv,
    VaspToTHz,
    kb,
    kJ_mol_to_eV,
)


def get_free_energy_classical(
    frequency: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    """
    Calculate the classical free energy.

    Args:
        frequency (np.ndarray): Array of frequencies.
        temperature (np.ndarray): Array of temperatures.

    Returns:
        np.ndarray: Array of classical free energies.
    """
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
    Returns thermal properties at constant volume in the given temperature range. Can only be called after job
    successfully ran.

    Args:
        eng_internal_dict (dict): Dictionary of internal energies for different strains.
        phonopy_dict (dict): Dictionary of Phonopy objects for different strains.
        structure_dict (dict): Dictionary of structures for different strains.
        repeat_vector (np.ndarray): Array of repeat vectors.
        fit_type (str): Type of fitting for energy-volume curve.
        fit_order (int): Order of fitting for energy-volume curve.
        t_min (float, optional): Minimum sample temperature. Defaults to 1.0.
        t_max (float, optional): Maximum sample temperature. Defaults to 1500.0.
        t_step (float, optional): Temperature sample interval. Defaults to 50.0.
        temperatures (np.ndarray, optional): Custom array of temperature samples. If given, t_min, t_max, t_step are ignored.
        cutoff_frequency (float, optional): Cutoff frequency. Defaults to None.
        pretend_real (bool, optional): Whether to pretend real. Defaults to False.
        band_indices (np.ndarray, optional): Array of band indices. Defaults to None.
        is_projection (bool, optional): Whether it is a projection. Defaults to False.
        quantum_mechanical (bool, optional): Whether to use quantum mechanical approach. Defaults to True.
        output_keys (tuple[str], optional): Output keys. Defaults to OutputThermodynamic.keys().

    Returns:
        dict: Thermal properties as returned by Phonopy.
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
    for i, _temp in enumerate(temperatures):
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
    Returns thermal properties at constant volume in the given temperature range. Can only be called after job
    successfully ran.

    Args:
        phonopy_dict (dict): Dictionary of Phonopy objects for different strains.
        t_min (float, optional): Minimum sample temperature. Defaults to 1.0.
        t_max (float, optional): Maximum sample temperature. Defaults to 1500.0.
        t_step (float, optional): Temperature sample interval. Defaults to 50.0.
        temperatures (np.ndarray, optional): Custom array of temperature samples. If given, t_min, t_max, t_step are ignored.
        cutoff_frequency (float, optional): Cutoff frequency. Defaults to None.
        pretend_real (bool, optional): Whether to pretend real. Defaults to False.
        band_indices (np.ndarray, optional): Array of band indices. Defaults to None.
        is_projection (bool, optional): Whether it is a projection. Defaults to False.
        output_keys (tuple[str], optional): Output keys. Defaults to OutputThermodynamic.keys().

    Returns:
        dict: Thermal properties as returned by Phonopy.
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
    Returns thermal properties at constant volume in the given temperature range. Can only be called after job
    successfully ran.

    Args:
        phonopy_dict (dict): Dictionary of Phonopy objects for different strains.
        t_min (float, optional): Minimum sample temperature. Defaults to 1.0.
        t_max (float, optional): Maximum sample temperature. Defaults to 1500.0.
        t_step (float, optional): Temperature sample interval. Defaults to 50.0.
        temperatures (np.ndarray, optional): Custom array of temperature samples. If given, t_min, t_max, t_step are ignored.
        cutoff_frequency (float, optional): Cutoff frequency. Defaults to None.

    Returns:
        dict: Thermal properties as returned by Phonopy.
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
                freqs_ev = np.array(freqs) * THzToEv
                cond = freqs_ev > cutoff_frequency
                t_property += (
                    np.sum(
                        get_free_energy_classical(
                            frequency=freqs_ev[cond], temperature=t
                        )
                    )
                    * w
                )
            t_property_lst.append(t_property / np.sum(phono.mesh.weights) * EvTokJmol)
        tp_collect_dict[strain] = {
            "temperatures": temperatures,
            "free_energy": np.array(t_property_lst) * kJ_mol_to_eV,
        }
    return tp_collect_dict


class QuasiHarmonicThermalProperties:
    def __init__(
        self,
        temperatures: np.ndarray,
        thermal_properties_dict: dict,
        strain_lst: np.ndarray,
        volumes_lst: np.ndarray,
        volumes_selected_lst: np.ndarray,
    ):
        """
        Initialize the QuasiHarmonicThermalProperties object.

        Args:
            temperatures (np.ndarray): Array of temperatures.
            thermal_properties_dict (dict): Dictionary of thermal properties.
            strain_lst (np.ndarray): Array of strains.
            volumes_lst (np.ndarray): Array of volumes.
            volumes_selected_lst (np.ndarray): Array of selected volumes.
        """
        self._temperatures = temperatures
        self._thermal_properties_dict = thermal_properties_dict
        self._strain_lst = strain_lst
        self._volumes_lst = volumes_lst
        self._volumes_selected_lst = volumes_selected_lst

    def get_property(self, thermal_property: str) -> np.ndarray:
        """
        Get the specified thermal property.

        Args:
            thermal_property (str): The thermal property to retrieve.

        Returns:
            np.ndarray: Array of the specified thermal property.
        """
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
        """
        Get the free energy.

        Returns:
            np.ndarray: Array of free energies.
        """
        return self.get_property(thermal_property="free_energy")

    def temperatures(self) -> np.ndarray:
        """
        Get the array of temperatures.

        Returns:
            np.ndarray: Array of temperatures.
        """
        return self._temperatures

    def entropy(self) -> np.ndarray:
        """
        Get the entropy.

        Returns:
            np.ndarray: Array of entropies.
        """
        return self.get_property(thermal_property="entropy")

    def heat_capacity(self) -> np.ndarray:
        """
        Get the heat capacity.

        Returns:
            np.ndarray: Array of heat capacities.
        """
        return self.get_property(thermal_property="heat_capacity")

    def volumes(self) -> np.ndarray:
        """
        Get the array of volumes.

        Returns:
            np.ndarray: Array of volumes.
        """
        return self._volumes_selected_lst


def generate_structures_helper(
    structure: Atoms,
    vol_range: Optional[float] = None,
    num_points: Optional[int] = None,
    strain_lst: Optional[list[float]] = None,
    displacement: float = 0.01,
    number_of_snapshots: int = None,
    interaction_range: float = 10.0,
    factor: float = VaspToTHz,
) -> tuple[dict, np.ndarray, dict, dict]:
    """
    Generate structures for the QuasiHarmonicWorkflow.

    Args:
        structure (Atoms): The input structure.
        vol_range (float, optional): The range of volume strain. Defaults to None.
        num_points (int, optional): The number of volume strain points. Defaults to None.
        strain_lst (List[float], optional): The list of volume strains. Defaults to None.
        displacement (float, optional): The displacement for finite difference calculation. Defaults to 0.01.
        number_of_snapshots (int, optional): The number of snapshots for each structure. Defaults to None.
        interaction_range (float, optional): The interaction range for supercell generation. Defaults to 10.0.
        factor (float, optional): The conversion factor from Vasp to THz. Defaults to VaspToTHz.

    Returns:
        tuple[dict, np.ndarray, dict, dict]: A tuple containing the following:
            - phonopy_dict (dict): Dictionary of Phonopy objects for different strains.
            - repeat_vector (np.ndarray): Array representing the repeat vector for supercell generation.
            - structure_energy_dict (dict): Dictionary of structure energies for different strains.
            - structure_forces_dict (dict): Dictionary of structure forces for different strains.
    """
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
) -> tuple[dict, dict]:
    """
    Analyze structures using Phonopy.

    Args:
        phonopy_dict (dict): Dictionary of Phonopy objects for different strains.
        output_dict (dict): Dictionary of output data for different strains.
        dos_mesh (int, optional): Density of states mesh. Defaults to 20.
        number_of_snapshots (int, optional): Number of snapshots. Defaults to None.
        output_keys (tuple[str], optional): Keys to include in the output dictionary. Defaults to ("force_constants", "mesh_dict").

    Returns:
        tuple[dict, dict]: A tuple containing the following:
            - eng_internal_dict (dict): Dictionary of internal energies for different strains.
            - phonopy_collect_dict (dict): Dictionary of Phonopy analysis results for different strains.
    """
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
        """
        Initialize the QuasiHarmonicWorkflow.

        Args:
            structure (Atoms): The input structure.
            num_points (int, optional): The number of points for the energy-volume curve. Defaults to 11.
            vol_range (float, optional): The range of volume strain. Defaults to 0.05.
            fit_type (str, optional): The type of fitting for the energy-volume curve. Defaults to "polynomial".
            fit_order (int, optional): The order of the fitting polynomial. Defaults to 3.
            interaction_range (float, optional): The interaction range for supercell generation. Defaults to 10.0.
            factor (float, optional): The conversion factor from Vasp to THz. Defaults to VaspToTHz.
            displacement (float, optional): The displacement for finite difference calculation. Defaults to 0.01.
            dos_mesh (int, optional): The density of states mesh. Defaults to 20.
            primitive_matrix (np.ndarray, optional): The primitive matrix for supercell generation. Defaults to None.
            number_of_snapshots (int, optional): The number of snapshots for each structure. Defaults to None.
        """
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
        """
        Generate structures for the QuasiHarmonicWorkflow.

        Returns:
            dict: A dictionary containing the calculated energies and forces for different strains.
        """
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
    ) -> tuple[dict, dict]:
        """
        Analyze structures using Phonopy.

        Args:
            output_dict (dict): Dictionary of output data for different strains.
            output_keys (tuple[str], optional): Keys to include in the output dictionary. Defaults to ("force_constants", "mesh_dict").

        Returns:
            tuple[dict, dict]: A tuple containing the following:
                - eng_internal_dict (dict): Dictionary of internal energies for different strains.
                - phonopy_collect_dict (dict): Dictionary of Phonopy analysis results for different strains.
        """
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
    ) -> dict:
        """
        Returns thermal properties at constant volume in the given temperature range. Can only be called after job
        successfully ran.

        Args:
            t_min (float): minimum sample temperature
            t_max (float): maximum sample temperature
            t_step (int):  temperature sample interval
            temperatures (array_like, float):  custom array of temperature samples, if given t_min, t_max, t_step are
                                               ignored.
            cutoff_frequency (float): cutoff frequency for phonon modes
            pretend_real (bool): if True, the real part of the phonon frequencies is returned
            band_indices (array_like, int): indices of bands to calculate
            is_projection (bool): if True, the phonon DOS is projected onto the band structure
            quantum_mechanical (bool): if True, the quantum mechanical partition function is used
            output_keys (tuple[str]): keys to include in the output dictionary

        Returns:
            Thermal: thermal properties as returned by Phonopy
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
            output_keys=output_keys,
        )
