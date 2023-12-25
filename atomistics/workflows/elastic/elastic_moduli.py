import numpy as np


def get_bulkmodul_voigt(C):
    return (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9


def get_shearmodul_voigt(C):
    return (
        (C[0, 0] + C[1, 1] + C[2, 2])
        - (C[0, 1] + C[0, 2] + C[1, 2])
        + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15


def get_youngsmodul_voigt(BV, GV):
    return (9 * BV * GV) / (3 * BV + GV)


def get_poissonsratio_voigt(BV, GV):
    return (1.5 * BV - GV) / (3 * BV + GV)


def get_bulkmodul_reuss(S):
    return 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))


def get_shearmodul_reuss(S):
    return 15 / (
        4 * (S[0, 0] + S[1, 1] + S[2, 2])
        - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
        + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
    )


def get_youngsmodul_reuss(BR, GR):
    return (9 * BR * GR) / (3 * BR + GR)


def get_poissonsratio_reuss(BR, GR):
    return (1.5 * BR - GR) / (3 * BR + GR)


def get_bulkmodul_hill(BV, BR):
    return _hill_approximation(voigt=BV, reuss=BR)


def get_shearmodul_hill(GV, GR):
    return _hill_approximation(voigt=GV, reuss=GR)


def get_youngsmodul_hill(BH, GH):
    return (9.0 * BH * GH) / (3.0 * BH + GH)


def get_poissonsratio_hill(BH, GH):
    return (1.5 * BH - GH) / (3.0 * BH + GH)


def get_AVR(GV, GR):
    return 100.0 * (GV - GR) / (GV + GR)


def get_elastic_matrix_eigval(C):
    return np.linalg.eig(C)


def get_elastic_matrix_inverse(C):
    return np.linalg.inv(C)


def _hill_approximation(voigt, reuss):
    return 0.50 * (voigt + reuss)
