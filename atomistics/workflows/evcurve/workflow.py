from collections import OrderedDict

import numpy as np
from ase.atoms import Atoms

from atomistics.shared.output import OutputEnergyVolumeCurve
from atomistics.workflows.evcurve.debye import (
    OutputThermodynamic,
    get_thermal_properties_for_energy_volume_curve,
)
from atomistics.workflows.evcurve.helper import (
    analyse_results_for_energy_volume_curve,
    get_tasks_for_energy_volume_curve,
)
from atomistics.workflows.interface import Workflow


class EnergyVolumeCurveWorkflow(Workflow):
    def __init__(
        self,
        structure: Atoms,
        num_points: int = 11,
        fit_type: str = "polynomial",
        fit_order: int = 3,
        vol_range: float = 0.05,
        axes: tuple[str, str, str] = ("x", "y", "z"),
        strains: list = None,
    ):
        """
        Initialize the EnergyVolumeCurveWorkflow object.

        Args:
            structure (Atoms): The atomic structure.
            num_points (int, optional): The number of points in the energy-volume curve. Defaults to 11.
            fit_type (str, optional): The type of fitting function. Defaults to "polynomial".
            fit_order (int, optional): The order of the fitting function. Defaults to 3.
            vol_range (float, optional): The range of volume variation. Defaults to 0.05.
            axes (tuple[str, str, str], optional): The axes along which to vary the volume. Defaults to ("x", "y", "z").
            strains (list, optional): The list of strains to apply. Defaults to None.
        """
        self.structure = structure
        self.num_points = num_points
        self.fit_type = fit_type
        self.vol_range = vol_range
        self.fit_order = fit_order
        self.axes = axes
        self.strains = strains
        self._task_dict = OrderedDict()
        self._fit_dict = {}

    @property
    def fit_dict(self) -> dict:
        """
        Get the fit dictionary.

        Returns:
            dict: The fit dictionary.
        """
        return self._fit_dict

    def generate_structures(self) -> dict:
        """
        Generate the structures for the energy-volume curve.

        Returns:
            dict: The generated structures.
        """
        self._task_dict = get_tasks_for_energy_volume_curve(
            structure=self.structure,
            vol_range=self.vol_range,
            num_points=self.num_points,
            strain_lst=self.strains,
            axes=self.axes,
        )
        return self._task_dict

    def analyse_structures(
        self, output_dict: dict, output_keys: tuple = OutputEnergyVolumeCurve.keys()
    ) -> dict:
        """
        Analyse the structures and fit the energy-volume curve.

        Args:
            output_dict (dict): The output dictionary.
            output_keys (tuple, optional): The keys to include in the output. Defaults to OutputEnergyVolumeCurve.keys().

        Returns:
            dict: The fit dictionary.
        """
        self._fit_dict = analyse_results_for_energy_volume_curve(
            output_dict=output_dict,
            task_dict=self._task_dict,
            fit_type=self.fit_type,
            fit_order=self.fit_order,
            output_keys=output_keys,
        )
        return self.fit_dict

    def get_thermal_properties(
        self,
        t_min: float = 1.0,
        t_max: float = 1500.0,
        t_step: float = 50.0,
        temperatures: np.ndarray = None,
        constant_volume: bool = False,
        output_keys: tuple[str] = OutputThermodynamic.keys(),
    ) -> dict:
        """
        Get the thermal properties of the system.

        Args:
            t_min (float, optional): The minimum temperature. Defaults to 1.0.
            t_max (float, optional): The maximum temperature. Defaults to 1500.0.
            t_step (float, optional): The temperature step. Defaults to 50.0.
            temperatures (np.ndarray, optional): The array of temperatures. Defaults to None.
            constant_volume (bool, optional): Whether to calculate properties at constant volume. Defaults to False.
            output_keys (tuple[str], optional): The keys to include in the output. Defaults to OutputThermodynamic.keys().

        Returns:
            dict: The thermal properties.
        """
        return get_thermal_properties_for_energy_volume_curve(
            fit_dict=self.fit_dict,
            masses=self.structure.get_masses(),
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            constant_volume=constant_volume,
            output_keys=output_keys,
        )
