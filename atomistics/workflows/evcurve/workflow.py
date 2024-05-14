import numpy as np
from ase.atoms import Atoms
from collections import OrderedDict

from atomistics.shared.output import OutputEnergyVolumeCurve
from atomistics.workflows.interface import Workflow
from atomistics.workflows.evcurve.debye import (
    get_thermal_properties,
    OutputThermodynamic,
)
from atomistics.workflows.evcurve.helper import (
    generate_structures_helper,
    analyse_structures_helper,
)


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
        self.structure = structure
        self.num_points = num_points
        self.fit_type = fit_type
        self.vol_range = vol_range
        self.fit_order = fit_order
        self.axes = axes
        self.strains = strains
        self._structure_dict = OrderedDict()
        self._fit_dict = {}

    @property
    def fit_dict(self) -> dict:
        return self._fit_dict

    def generate_structures(self) -> dict:
        """

        Returns:
            (dict)
        """
        self._structure_dict = OrderedDict(
            generate_structures_helper(
                structure=self.structure,
                vol_range=self.vol_range,
                num_points=self.num_points,
                strain_lst=self.strains,
                axes=self.axes,
            )
        )
        return {"calc_energy": self._structure_dict}

    def analyse_structures(
        self, output_dict: dict, output_keys: tuple = OutputEnergyVolumeCurve.keys()
    ) -> dict:
        self._fit_dict = analyse_structures_helper(
            output_dict=output_dict,
            structure_dict=self._structure_dict,
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
        return get_thermal_properties(
            fit_dict=self.fit_dict,
            masses=self.structure.get_masses(),
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            constant_volume=constant_volume,
            output_keys=output_keys,
        )
