import numpy as np


def get_BV(C):
    return (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9


def get_GV(C):
    return (
        (C[0, 0] + C[1, 1] + C[2, 2])
        - (C[0, 1] + C[0, 2] + C[1, 2])
        + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15


def get_EV(BV, GV):
    return (9 * BV * GV) / (3 * BV + GV)


def get_nuV(BV, GV):
    return (1.5 * BV - GV) / (3 * BV + GV)


def get_BR(S):
    return 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))


def get_GR(S):
    return 15 / (
        4 * (S[0, 0] + S[1, 1] + S[2, 2])
        - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
        + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
    )


def get_ER(BR, GR):
    return (9 * BR * GR) / (3 * BR + GR)


def get_nuR(BR, GR):
    return (1.5 * BR - GR) / (3 * BR + GR)


def get_BH(BV, BR):
    return 0.50 * (BV + BR)


def get_GH(GV, GR):
    return 0.50 * (GV + GR)


def get_EH(BH, GH):
    return (9.0 * BH * GH) / (3.0 * BH + GH)


def get_nuH(BH, GH):
    return (1.5 * BH - GH) / (3.0 * BH + GH)


def get_AVR(GV, GR):
    return 100.0 * (GV - GR) / (GV + GR)


def get_C_eigval(C):
    return np.linalg.eig(C)


def get_S(C):
    return np.linalg.inv(C)
