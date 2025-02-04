from collections import OrderedDict
from typing import Any, Optional

import ase.atoms
import numpy as np
import scipy.constants

from atomistics.shared.output import OutputElastic
from atomistics.workflows.elastic.elastic_moduli import ElasticProperties
from atomistics.workflows.elastic.symmetry import (
    Ls_Dic,
    get_C_from_A2,
    symmetry_analysis,
)


def generate_structures_helper(
    structure: ase.atoms.Atoms,
    eps_range: float,
    num_of_point: int,
    zero_strain_job_name: str = "s_e_0",
    sqrt_eta: bool = True,
) -> tuple[dict[str, int], dict[str, ase.atoms.Atoms]]:
    """
    Generate structures for elastic analysis.

    Args:
        structure (ase.atoms.Atoms): The input structure.
        eps_range (float): The range of strain.
        num_of_point (int): The number of points in the strain range.
        zero_strain_job_name (str, optional): The name of the zero strain job. Defaults to "s_e_0".
        sqrt_eta (bool, optional): Whether to take the square root of the eta matrix. Defaults to True.

    Returns:
        Tuple[Dict[str, int], Dict[str, ase.atoms.Atoms]]: A tuple containing the symmetry dictionary and the structure dictionary.
    """
    SGN, v0, LC, Lag_strain_list, epss = symmetry_analysis(
        structure=structure,
        eps_range=eps_range,
        num_of_point=num_of_point,
    )
    sym_dict = {
        "SGN": SGN,
        "v0": v0,
        "LC": LC,
        "Lag_strain_list": Lag_strain_list,
        "epss": epss,
    }

    structure_dict = OrderedDict()
    if 0.0 in epss:
        structure_dict[zero_strain_job_name] = structure.copy()

    for lag_strain in Lag_strain_list:
        Ls_list = Ls_Dic[lag_strain]
        for eps in epss:
            if eps == 0.0:
                continue

            Ls = np.zeros(6)
            for ii in range(6):
                Ls[ii] = Ls_list[ii]
            Lv = eps * Ls

            eta_matrix = np.zeros((3, 3))

            eta_matrix[0, 0] = Lv[0]
            eta_matrix[0, 1] = Lv[5] / 2.0
            eta_matrix[0, 2] = Lv[4] / 2.0

            eta_matrix[1, 0] = Lv[5] / 2.0
            eta_matrix[1, 1] = Lv[1]
            eta_matrix[1, 2] = Lv[3] / 2.0

            eta_matrix[2, 0] = Lv[4] / 2.0
            eta_matrix[2, 1] = Lv[3] / 2.0
            eta_matrix[2, 2] = Lv[2]

            norm = 1.0
            eps_matrix = eta_matrix
            if np.linalg.norm(eta_matrix) > 0.7:
                raise Exception(f"Too large deformation {eps:g}")

            if sqrt_eta:
                while norm > 1.0e-10:
                    x = eta_matrix - np.dot(eps_matrix, eps_matrix) / 2.0
                    norm = np.linalg.norm(x - eps_matrix)
                    eps_matrix = x

            # --- Calculating the M_new matrix ---------------------------------------------------------
            i_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            def_matrix = i_matrix + eps_matrix
            scell = np.dot(structure.get_cell(), def_matrix)
            nstruct = structure.copy()
            nstruct.set_cell(scell, scale_atoms=True)

            structure_dict[_subjob_name(i=lag_strain, eps=eps)] = nstruct

    return sym_dict, structure_dict


def analyse_structures_helper(
    output_dict: dict,
    sym_dict: dict,
    fit_order: int = 2,
    zero_strain_job_name: str = "s_e_0",
    output_keys: tuple = OutputElastic.keys(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Analyze structures and calculate elastic properties.

    Args:
        output_dict (dict): The dictionary containing the output data.
        sym_dict (dict): The symmetry dictionary.
        fit_order (int, optional): The order of the polynomial fit. Defaults to 2.
        zero_strain_job_name (str, optional): The name of the zero strain job. Defaults to "s_e_0".
        output_keys (tuple, optional): The keys to include in the output dictionary. Defaults to OutputElastic.keys().

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the updated symmetry dictionary and the elastic properties dictionary.
    """
    elastic_matrix, A2, strain_energy, ene0 = _get_elastic_matrix(
        output_dict=output_dict,
        Lag_strain_list=sym_dict["Lag_strain_list"],
        epss=sym_dict["epss"],
        v0=sym_dict["v0"],
        LC=sym_dict["LC"],
        fit_order=fit_order,
        zero_strain_job_name=zero_strain_job_name,
    )
    sym_dict["strain_energy"] = strain_energy
    sym_dict["e0"] = ene0
    sym_dict["A2"] = A2
    return sym_dict, ElasticProperties(elastic_matrix=elastic_matrix).to_dict(
        output_keys=output_keys
    )


def _get_elastic_matrix(
    output_dict: dict,
    Lag_strain_list: list[float],
    epss: float,
    v0: float,
    LC: str,
    fit_order: int = 2,
    zero_strain_job_name: str = "s_e_0",
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[float, float]]], Optional[float]]:
    """
    Calculate the elastic matrix and other properties.

    Args:
        output_dict (dict): The dictionary containing the output data.
        Lag_strain_list (list[float]): The list of Lagrangian strains.
        epss (float): The list of strains.
        v0 (float): The volume of the unit cell.
        LC (str): The lattice constant.
        fit_order (int, optional): The order of the polynomial fit. Defaults to 2.
        zero_strain_job_name (str, optional): The name of the zero strain job. Defaults to "s_e_0".

    Returns:
        Tuple[np.ndarray, np.ndarray, List[List[Tuple[float, float]]], Optional[float]]: A tuple containing the elastic matrix, A2 coefficients, strain energy data, and ene0 value.
    """
    if "energy" in output_dict:
        output_dict = output_dict["energy"]

    ene0 = None
    if 0.0 in epss:
        ene0 = output_dict[zero_strain_job_name]
    strain_energy = []
    for lag_strain in Lag_strain_list:
        strain_energy.append([])
        for eps in epss:
            if eps != 0.0:
                ene = output_dict[_subjob_name(i=lag_strain, eps=eps)]
            else:
                ene = ene0
            strain_energy[-1].append((eps, ene))
    elastic_matrix, A2 = _fit_elastic_matrix(
        strain_ene=strain_energy,
        v0=v0,
        LC=LC,
        fit_order=int(fit_order),
    )
    return elastic_matrix, A2, strain_energy, ene0


def _subjob_name(i: int, eps: float) -> str:
    """
    Generate the subjob name.

    Args:
        i (int): The index.
        eps (float): The strain value.

    Returns:
        str: The subjob name.
    """
    return (f"s_{i}_e_{eps:.5f}").replace(".", "_").replace("-", "m")


def _fit_elastic_matrix(
    strain_ene: list[list[tuple[float, float]]], v0: float, LC: str, fit_order: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the elastic matrix from strain-energy data.

    Args:
        strain_ene (list[list[tuple[float, float]]]): The strain-energy data.
        v0 (float): The volume of the unit cell.
        LC (str): The lattice constant.
        fit_order (int): The order of the polynomial fit.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the elastic matrix and the A2 coefficients.
    """
    A2 = []
    for s_e in strain_ene:
        ss = np.transpose(s_e)
        coeffs = np.polyfit(ss[0], ss[1] / v0, fit_order)
        A2.append(coeffs[fit_order - 2])

    A2 = np.array(A2)
    C = get_C_from_A2(A2, LC)

    for i in range(5):
        for j in range(i + 1, 6):
            C[j, i] = C[i, j]

    CONV = (
        1e21 / scipy.constants.physical_constants["joule-electron volt relationship"][0]
    )  # From eV/Ang^3 to GPa

    C *= CONV
    return C, A2
