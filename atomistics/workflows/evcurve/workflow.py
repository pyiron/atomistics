import numpy as np
from ase.atoms import Atoms
from typing import Literal
from collections import OrderedDict

from atomistics.workflows.evcurve.fit import EnergyVolumeFit
from atomistics.workflows.shared.workflow import Workflow


def _strain_axes(
    structure: Atoms, axes: Literal["x", "y", "z"], volume_strain: float
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


class EnergyVolumeCurveWorkflow(Workflow):
    def __init__(
        self,
        structure,
        num_points=11,
        fit_type="polynomial",
        fit_order=3,
        vol_range=0.05,
        axes=["x", "y", "z"],
        strains=None,
    ):
        self.structure = structure
        self.num_points = num_points
        self.fit_type = fit_type
        self.vol_range = vol_range
        self.fit_order = fit_order
        self.axes = axes
        self.strains = strains
        self.fit_module = EnergyVolumeFit()
        self._structure_dict = OrderedDict()

    @property
    def fit_dict(self):
        return self.fit_module.fit_dict

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
            basis = _strain_axes(self.structure, self.axes, strain)
            self._structure_dict[1 + np.round(strain, 7)] = basis
        return {"calc_energy": self._structure_dict}

    def analyse_structures(self, output_dict):
        if "energy" in output_dict.keys():
            output_dict = output_dict["energy"]
        self.fit_module = EnergyVolumeFit(
            volume_lst=self.get_volume_lst(),
            energy_lst=[output_dict[k] for k in self._structure_dict.keys()],
        )
        self.fit_module.fit(fit_type=self.fit_type, fit_order=self.fit_order)
        return self.fit_dict

    def get_volume_lst(self):
        return [
            self._structure_dict[k].get_volume() for k in self._structure_dict.keys()
        ]
