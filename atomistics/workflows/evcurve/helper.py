import numpy as np
from ase.atoms import Atoms
from typing import Optional, List

from atomistics.shared.output import OutputEnergyVolumeCurve
from atomistics.workflows.evcurve.fit import EnergyVolumeFit


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


def apply_strain(
    structure: Atoms, epsilon: float, return_box: bool = False, mode: str = "linear"
) -> Atoms:
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


def get_energy_lst(output_dict: dict, structure_dict: dict) -> list:
    return [output_dict["energy"][k] for k in structure_dict.keys()]


def get_volume_lst(structure_dict: dict) -> list:
    return [structure.get_volume() for structure in structure_dict.values()]


def fit_ev_curve_internal(
    volume_lst: np.ndarray, energy_lst: np.ndarray, fit_type: str, fit_order: int
) -> EnergyVolumeFit:
    fit_module = EnergyVolumeFit(
        volume_lst=volume_lst,
        energy_lst=energy_lst,
    )
    fit_module.fit(fit_type=fit_type, fit_order=fit_order)
    return fit_module


def fit_ev_curve(
    volume_lst: np.ndarray, energy_lst: np.ndarray, fit_type: str, fit_order: int
) -> dict:
    return fit_ev_curve_internal(
        volume_lst=volume_lst,
        energy_lst=energy_lst,
        fit_type=fit_type,
        fit_order=fit_order,
    ).fit_dict


def get_strains(
    vol_range: Optional[float] = None,
    num_points: Optional[int] = None,
    strain_lst: Optional[List[float]] = None,
) -> np.ndarray:
    if strain_lst is None and (vol_range is None or num_points is None):
        raise ValueError(
            "Either the strain_lst parameter or the vol_range and the num_points parameter have to be defined."
        )
    elif strain_lst is None and vol_range is not None and num_points is not None:
        return np.linspace(
            -vol_range,
            vol_range,
            int(num_points),
        )
    else:
        return strain_lst


def generate_structures_helper(
    structure: Atoms,
    vol_range: Optional[float] = None,
    num_points: Optional[int] = None,
    strain_lst: Optional[List[float]] = None,
    axes: tuple[str, str, str] = ("x", "y", "z"),
) -> dict:
    strain_lst = get_strains(
        vol_range=vol_range, num_points=num_points, strain_lst=strain_lst
    )
    key_lst = [1 + np.round(strain, 7) for strain in strain_lst]
    value_lst = [
        _strain_axes(structure=structure, axes=axes, volume_strain=strain)
        for strain in strain_lst
    ]
    return {key: value for key, value in zip(key_lst, value_lst)}


def analyse_structures_helper(
    output_dict: dict,
    structure_dict: dict,
    fit_type: str = "polynomial",
    fit_order: int = 3,
    output_keys: tuple = OutputEnergyVolumeCurve.keys(),
) -> dict:
    return EnergyVolumeCurveProperties(
        fit_module=fit_ev_curve_internal(
            volume_lst=get_volume_lst(structure_dict=structure_dict),
            energy_lst=get_energy_lst(
                output_dict=output_dict, structure_dict=structure_dict
            ),
            fit_type=fit_type,
            fit_order=fit_order,
        )
    ).to_dict(output_keys=output_keys)


class EnergyVolumeCurveProperties:
    def __init__(self, fit_module):
        self._fit_module = fit_module

    def volume_eq(self) -> float:
        return self._fit_module.fit_dict["volume_eq"]

    def energy_eq(self) -> float:
        return self._fit_module.fit_dict["energy_eq"]

    def bulkmodul_eq(self) -> float:
        return self._fit_module.fit_dict["bulkmodul_eq"]

    def b_prime_eq(self) -> float:
        return self._fit_module.fit_dict["b_prime_eq"]

    def volume(self) -> np.ndarray:
        return self._fit_module.fit_dict["volume"]

    def energy(self) -> np.ndarray:
        return self._fit_module.fit_dict["energy"]

    def fit_dict(self) -> dict:
        return {
            k: self._fit_module.fit_dict[k]
            for k in ["fit_type", "least_square_error", "poly_fit", "fit_order"]
            if k in self._fit_module.fit_dict.keys()
        }

    def to_dict(self, output_keys: tuple = OutputEnergyVolumeCurve.keys()) -> dict:
        return OutputEnergyVolumeCurve(
            **{k: getattr(self, k) for k in OutputEnergyVolumeCurve.keys()}
        ).get(output_keys=output_keys)
