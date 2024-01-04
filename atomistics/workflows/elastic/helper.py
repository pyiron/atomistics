from collections import OrderedDict

import numpy as np
import scipy.constants

from atomistics.workflows.elastic.symmetry import (
    symmetry_analysis,
    get_C_from_A2,
    Ls_Dic,
)


def generate_structures_helper(
    structure, eps_range, num_of_point, zero_strain_job_name="s_e_0", sqrt_eta=True
):
    """

    Returns:

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
                raise Exception("Too large deformation %g" % eps)

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
    output_dict,
    Lag_strain_list,
    epss,
    v0,
    LC,
    fit_order=2,
    zero_strain_job_name="s_e_0",
):
    """

    Args:
        output_dict (dict):
        output (tuple):

    Returns:

    """
    if "energy" in output_dict.keys():
        output_dict = output_dict["energy"]

    ene0 = None
    if 0.0 in epss:
        ene0 = output_dict[zero_strain_job_name]
    strain_energy = []
    for lag_strain in Lag_strain_list:
        strain_energy.append([])
        for eps in epss:
            if not eps == 0.0:
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


def _subjob_name(i, eps):
    """

    Args:
        i:
        eps:

    Returns:

    """
    return ("s_%s_e_%.5f" % (i, eps)).replace(".", "_").replace("-", "m")


def _fit_elastic_matrix(strain_ene, v0, LC, fit_order):
    """

    Returns:

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
