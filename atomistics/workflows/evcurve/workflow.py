import numpy as np
from ase.atoms import Atoms
from collections import OrderedDict

from atomistics.shared.output import OutputEnergyVolumeCurve
from atomistics.workflows.evcurve.fit import EnergyVolumeFit
from atomistics.workflows.interface import Workflow
from atomistics.workflows.evcurve.debye import (
    get_thermal_properties,
    OutputThermodynamic,
)


def _strain_axes(
    structure: Atoms, volume_strain: float, axes: tuple[str, str, str] = ("x", "y", "z")
) -> Atoms:
    """
    Strain box along given axes to achieve given *volumetric* strain.

    Returns a copy.
    """
    axes = np.array([a in axes for a in ("x", "y", "z")])
    num_axes = sum(axes)
    # formula calculates the strain along each axis to achieve the overall volumetric strain
    # beware that: (1+e)**x - 1 != e**x
    strains = axes * ((1 + volume_strain) ** (1.0 / num_axes) - 1)
    return apply_strain(structure=structure, epsilon=strains, return_box=True)


def apply_strain(structure, epsilon, return_box=False, mode="linear"):
    """
    Apply a given strain on the structure. It applies the matrix `F` in the manner:

    ```
        new_cell = F @ current_cell
    ```

    Args:
        epsilon (float/list/ndarray): epsilon matrix. If a single number is set, the same
            strain is applied in each direction. If a 3-dim vector is set, it will be
            multiplied by a unit matrix.
        return_box (bool): whether to return a box. If set to True, only the returned box will
            have the desired strain and the original box will stay unchanged.
        mode (str): `linear` or `lagrangian`. If `linear`, `F` is equal to the epsilon - 1.
            If `lagrangian`, epsilon is given by `(F^T * F - 1) / 2`. It raises an error if
            the strain is not symmetric (if the shear components are given).
    """
    epsilon = np.array([epsilon]).flatten()
    if len(epsilon) == 3 or len(epsilon) == 1:
        epsilon = epsilon * np.eye(3)
    epsilon = epsilon.reshape(3, 3)
    if epsilon.min() < -1.0:
        raise ValueError("Strain value too negative")
    if return_box:
        structure_copy = structure.copy()
    else:
        structure_copy = structure
    cell = structure_copy.cell.copy()
    if mode == "linear":
        F = epsilon + np.eye(3)
    elif mode == "lagrangian":
        if not np.allclose(epsilon, epsilon.T):
            raise ValueError("Strain must be symmetric if `mode = 'lagrangian'`")
        E, V = np.linalg.eigh(2 * epsilon + np.eye(3))
        F = np.einsum("ik,k,jk->ij", V, np.sqrt(E), V)
    else:
        raise ValueError("mode must be `linear` or `lagrangian`")
    cell = np.matmul(F, cell)
    structure_copy.set_cell(cell, scale_atoms=True)
    if return_box:
        return structure_copy


def get_energy_lst(output_dict, structure_dict):
    return [output_dict["energy"][k] for k in structure_dict.keys()]


def get_volume_lst(structure_dict):
    return [structure.get_volume() for structure in structure_dict.values()]


def fit_ev_curve_internal(volume_lst, energy_lst, fit_type, fit_order):
    fit_module = EnergyVolumeFit(
        volume_lst=volume_lst,
        energy_lst=energy_lst,
    )
    fit_module.fit(fit_type=fit_type, fit_order=fit_order)
    return fit_module


def fit_ev_curve(volume_lst, energy_lst, fit_type, fit_order):
    return fit_ev_curve_internal(
        volume_lst=volume_lst,
        energy_lst=energy_lst,
        fit_type=fit_type,
        fit_order=fit_order,
    ).fit_dict


class EnergyVolumeCurveProperties:
    def __init__(self, fit_module):
        self._fit_module = fit_module

    def get_volume_eq(self):
        return self._fit_module.fit_dict["volume_eq"]

    def get_energy_eq(self):
        return self._fit_module.fit_dict["energy_eq"]

    def get_bulkmodul_eq(self):
        return self._fit_module.fit_dict["bulkmodul_eq"]

    def get_bulkmodul_pressure_derivative_eq(self):
        return self._fit_module.fit_dict["b_prime_eq"]

    def get_volumes(self):
        return self._fit_module.fit_dict["volume"]

    def get_energies(self):
        return self._fit_module.fit_dict["energy"]

    def get_fit_dict(self):
        return {
            k: self._fit_module.fit_dict[k]
            for k in ["fit_type", "least_square_error", "poly_fit", "fit_order"]
            if k in self._fit_module.fit_dict.keys()
        }


EnergyVolumeCurveOutputEnergyVolumeCurve = OutputEnergyVolumeCurve(
    fit_dict=EnergyVolumeCurveProperties.get_fit_dict,
    energy=EnergyVolumeCurveProperties.get_energies,
    volume=EnergyVolumeCurveProperties.get_volumes,
    b_prime_eq=EnergyVolumeCurveProperties.get_bulkmodul_pressure_derivative_eq,
    bulkmodul_eq=EnergyVolumeCurveProperties.get_bulkmodul_eq,
    energy_eq=EnergyVolumeCurveProperties.get_energy_eq,
    volume_eq=EnergyVolumeCurveProperties.get_volume_eq,
)


class EnergyVolumeCurveWorkflow(Workflow):
    def __init__(
        self,
        structure,
        num_points=11,
        fit_type="polynomial",
        fit_order=3,
        vol_range=0.05,
        axes=("x", "y", "z"),
        strains=None,
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
    def fit_dict(self):
        return self._fit_dict

    def generate_structures(self):
        """

        Returns:
            (dict)
        """
        strains = self.strains
        if strains is None:
            strains = np.linspace(
                -self.vol_range,
                self.vol_range,
                int(self.num_points),
            )
        for strain in strains:
            basis = _strain_axes(
                structure=self.structure, axes=self.axes, volume_strain=strain
            )
            self._structure_dict[1 + np.round(strain, 7)] = basis
        return {"calc_energy": self._structure_dict}

    def analyse_structures(self, output_dict, output=OutputEnergyVolumeCurve.fields()):
        self._fit_dict = EnergyVolumeCurveOutputEnergyVolumeCurve.get(
            EnergyVolumeCurveProperties(
                fit_module=fit_ev_curve_internal(
                    volume_lst=get_volume_lst(structure_dict=self._structure_dict),
                    energy_lst=get_energy_lst(
                        output_dict=output_dict, structure_dict=self._structure_dict
                    ),
                    fit_type=self.fit_type,
                    fit_order=self.fit_order,
                )
            ),
            *output,
        )
        return self.fit_dict

    def get_volume_lst(self):
        return get_volume_lst(structure_dict=self._structure_dict)

    def get_thermal_expansion(
        self, output_dict, t_min=1, t_max=1500, t_step=50, temperatures=None
    ):
        self.analyse_structures(output_dict=output_dict)
        thermal_properties_dict = get_thermal_properties(
            fit_dict=self.fit_dict,
            masses=self.structure.get_masses(),
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            constant_volume=False,
            output=["temperatures", "volumes"],
        )
        return (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )

    def get_thermal_properties(
        self,
        t_min=1,
        t_max=1500,
        t_step=50,
        temperatures=None,
        constant_volume=False,
        output=OutputThermodynamic.fields(),
    ):
        return get_thermal_properties(
            fit_dict=self.fit_dict,
            masses=self.structure.get_masses(),
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            constant_volume=constant_volume,
            output=output,
        )
