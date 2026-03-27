from typing import Optional, Union

import numpy as np
from ase.atoms import Atoms

from atomistics.shared.output import OutputEnergyVolumeCurve
from atomistics.workflows.evcurve.fit import EnergyVolumeFit


def _strain_axes(
    structure: Atoms, volume_strain: float, axes: tuple[str, str, str] = ("x", "y", "z")
) -> Atoms:
    """
    Strain the box along the given axes to achieve the given volumetric strain.

    Args:
        structure (Atoms): The input structure.
        volume_strain (float): The desired volumetric strain.
        axes (tuple[str, str, str], optional): The axes along which to strain the box. Defaults to ("x", "y", "z").

    Returns:
        Atoms: The strained structure.

    Raises:
        ValueError: If the number of axes is zero.

    """
    axes = np.array([a in axes for a in ("x", "y", "z")])
    num_axes = sum(axes)
    if num_axes == 0:
        raise ValueError("At least one axis must be selected.")
    # Formula calculates the strain along each axis to achieve the overall volumetric strain.
    # Beware that: (1+e)**x - 1 != e**x
    strains = axes * ((1 + volume_strain) ** (1.0 / num_axes) - 1)
    return apply_strain(structure=structure, epsilon=strains, return_box=True)


def apply_strain(
    structure: Atoms,
    epsilon: Union[float, list[float], np.ndarray],
    return_box: bool = False,
    mode: str = "linear",
) -> Atoms:
    """
    Apply a given strain on the structure. It applies the matrix `F` in the manner:

    ```
        new_cell = F @ current_cell
    ```

    Args:
        structure (Atoms): The input structure.
        epsilon (float/list/ndarray): The epsilon matrix. If a single number is set, the same
            strain is applied in each direction. If a 3-dim vector is set, it will be
            multiplied by a unit matrix.
        return_box (bool, optional): Whether to return a box. If set to True, only the returned box will
            have the desired strain and the original box will stay unchanged. Defaults to False.
        mode (str, optional): The mode of strain application. Can be 'linear' or 'lagrangian'.
            If 'linear', `F` is equal to the epsilon - 1. If 'lagrangian', epsilon is given by
            `(F^T * F - 1) / 2`. It raises an error if the strain is not symmetric (if the shear
            components are given). Defaults to 'linear'.

    Returns:
        Atoms: The structure with the applied strain.

    Raises:
        ValueError: If the strain value is too negative or if the strain is not symmetric in 'lagrangian' mode.

    """
    epsilon = np.array([epsilon]).flatten()
    if len(epsilon) == 3 or len(epsilon) == 1:
        epsilon = epsilon * np.eye(3)
    epsilon = epsilon.reshape(3, 3)
    if epsilon.min() < -1.0:
        raise ValueError("Strain value too negative")
    structure_copy = structure.copy() if return_box else structure
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


def get_energy_lst(output_dict: dict, structure_dict: dict) -> list[float]:
    """
    Get a list of energy values from the output dictionary for each structure in the structure dictionary.

    Args:
        output_dict (dict): The output dictionary containing energy values.
        structure_dict (dict): The structure dictionary containing structure keys.

    Returns:
        List[float]: A list of energy values.

    """
    return [output_dict["energy"][k] for k in structure_dict]


def get_volume_lst(structure_dict: dict) -> list[float]:
    """
    Get a list of volume values from the structure dictionary.

    Args:
        structure_dict (dict): The structure dictionary containing structure keys.

    Returns:
        List[float]: A list of volume values.

    """
    return [structure.get_volume() for structure in structure_dict.values()]


def fit_ev_curve_internal(
    volume_lst: np.ndarray, energy_lst: np.ndarray, fit_type: str, fit_order: int
) -> EnergyVolumeFit:
    """
    Fit an energy-volume curve using the given volume and energy arrays.

    Args:
        volume_lst (np.ndarray): The array of volume values.
        energy_lst (np.ndarray): The array of energy values.
        fit_type (str): The type of fit to perform. Can be 'polynomial' or 'birch_murnaghan'.
        fit_order (int): The order of the polynomial fit. Only applicable if fit_type is 'polynomial'.

    Returns:
        EnergyVolumeFit: The fitted energy-volume curve.

    """
    fit_module = EnergyVolumeFit(
        volume_lst=volume_lst,
        energy_lst=energy_lst,
    )
    fit_module.fit(fit_type=fit_type, fit_order=fit_order)
    return fit_module


def fit_ev_curve(
    volume_lst: np.ndarray, energy_lst: np.ndarray, fit_type: str, fit_order: int
) -> dict:
    """
    Fit an energy-volume curve using the given volume and energy arrays.

    Args:
        volume_lst (np.ndarray): The array of volume values.
        energy_lst (np.ndarray): The array of energy values.
        fit_type (str): The type of fit to perform. Can be 'polynomial' or 'birch_murnaghan'.
        fit_order (int): The order of the polynomial fit. Only applicable if fit_type is 'polynomial'.

    Returns:
        dict: The fitted energy-volume curve.

    """
    return fit_ev_curve_internal(
        volume_lst=volume_lst,
        energy_lst=energy_lst,
        fit_type=fit_type,
        fit_order=fit_order,
    ).fit_dict


def get_strains(
    vol_range: Optional[float] = None,
    num_points: Optional[int] = None,
    strain_lst: Optional[list[float]] = None,
) -> np.ndarray:
    """
    Generate an array of strain values.

    Args:
        vol_range (float, optional): The range of volumetric strain. Defaults to None.
        num_points (int, optional): The number of points to generate. Defaults to None.
        strain_lst (List[float], optional): A list of predefined strain values. Defaults to None.

    Returns:
        np.ndarray: An array of strain values.

    Raises:
        ValueError: If neither strain_lst nor vol_range and num_points are defined.

    """
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
        return np.array(strain_lst)


def get_tasks_for_energy_volume_curve(
    structure: Atoms,
    vol_range: Optional[float] = None,
    num_points: Optional[int] = None,
    strain_lst: Optional[list[float]] = None,
    axes: tuple[str, str, str] = ("x", "y", "z"),
) -> dict:
    """
    Generate a dictionary of strained structures.

    Args:
        structure (Atoms): The input structure.
        vol_range (float, optional): The range of volumetric strain. Defaults to None.
        num_points (int, optional): The number of points to generate. Defaults to None.
        strain_lst (List[float], optional): A list of predefined strain values. Defaults to None.
        axes (tuple[str, str, str], optional): The axes along which to strain the box. Defaults to ("x", "y", "z").

    Returns:
        dict: A dictionary of strained structures, where the keys are the strain values and the values are the strained structures.

    Raises:
        ValueError: If neither strain_lst nor vol_range and num_points are defined.

    """
    strain_lst = get_strains(
        vol_range=vol_range, num_points=num_points, strain_lst=strain_lst
    )
    key_lst = [1 + np.round(strain, 7) for strain in strain_lst]
    value_lst = [
        _strain_axes(structure=structure, axes=axes, volume_strain=strain)
        for strain in strain_lst
    ]
    return {"calc_energy": dict(zip(key_lst, value_lst))}


def analyse_results_for_energy_volume_curve(
    output_dict: dict,
    task_dict: dict,
    fit_type: str = "polynomial",
    fit_order: int = 3,
    output_keys: tuple = OutputEnergyVolumeCurve.keys(),
) -> dict:
    """
    Analyze structures using the output and structure dictionaries.

    Args:
        output_dict (dict): The output dictionary containing energy values.
        task_dict (dict): The structure dictionary containing structure keys.
        fit_type (str, optional): The type of fit to perform. Can be 'polynomial' or 'birch_murnaghan'.
            Defaults to 'polynomial'.
        fit_order (int, optional): The order of the polynomial fit. Only applicable if fit_type is 'polynomial'.
            Defaults to 3.
        output_keys (tuple, optional): The keys to include in the output dictionary. Defaults to OutputEnergyVolumeCurve.keys().

    Returns:
        dict: The analyzed structures.

    """
    return EnergyVolumeCurveProperties(
        fit_module=fit_ev_curve_internal(
            volume_lst=get_volume_lst(
                structure_dict=task_dict["calc_energy"],
            ),
            energy_lst=get_energy_lst(
                output_dict=output_dict,
                structure_dict=task_dict["calc_energy"],
            ),
            fit_type=fit_type,
            fit_order=fit_order,
        )
    ).to_dict(output_keys=output_keys)


class EnergyVolumeCurveProperties:
    def __init__(self, fit_module: EnergyVolumeFit) -> None:
        """
        Initialize the EnergyVolumeCurveProperties class.

        Args:
            fit_module (EnergyVolumeFit): The fitted energy-volume curve module.
        """
        self._fit_module = fit_module

    def volume_eq(self) -> float:
        """
        Get the equilibrium volume.

        Returns:
            float: The equilibrium volume.
        """
        return self._fit_module.fit_dict["volume_eq"]

    def energy_eq(self) -> float:
        """
        Get the equilibrium energy.

        Returns:
            float: The equilibrium energy.
        """
        return self._fit_module.fit_dict["energy_eq"]

    def bulkmodul_eq(self) -> float:
        """
        Get the equilibrium bulk modulus.

        Returns:
            float: The equilibrium bulk modulus.
        """
        return self._fit_module.fit_dict["bulkmodul_eq"]

    def b_prime_eq(self) -> float:
        """
        Get the equilibrium derivative of bulk modulus with respect to pressure.

        Returns:
            float: The equilibrium derivative of bulk modulus with respect to pressure.
        """
        return self._fit_module.fit_dict["b_prime_eq"]

    def volume(self) -> np.ndarray:
        """
        Get the array of volume values.

        Returns:
            np.ndarray: The array of volume values.
        """
        return self._fit_module.fit_dict["volume"]

    def energy(self) -> np.ndarray:
        """
        Get the array of energy values.

        Returns:
            np.ndarray: The array of energy values.
        """
        return self._fit_module.fit_dict["energy"]

    def fit_dict(self) -> dict:
        """
        Get the fit dictionary.

        Returns:
            dict: The fit dictionary.
        """
        return {
            k: self._fit_module.fit_dict[k]
            for k in ["fit_type", "least_square_error", "poly_fit", "fit_order"]
            if k in self._fit_module.fit_dict
        }

    def to_dict(self, output_keys: tuple = OutputEnergyVolumeCurve.keys()) -> dict:
        """
        Convert the EnergyVolumeCurveProperties object to a dictionary.

        Args:
            output_keys (tuple, optional): The keys to include in the output dictionary.
                Defaults to OutputEnergyVolumeCurve.keys().

        Returns:
            dict: The converted dictionary.
        """
        return OutputEnergyVolumeCurve(
            **{k: getattr(self, k) for k in OutputEnergyVolumeCurve.keys()}
        ).get(output_keys=output_keys)
