import numpy as np


def get_bulkmodul_voigt(elastic_matrix):
    return (
        elastic_matrix[0, 0]
        + elastic_matrix[1, 1]
        + elastic_matrix[2, 2]
        + 2 * (elastic_matrix[0, 1] + elastic_matrix[0, 2] + elastic_matrix[1, 2])
    ) / 9


def get_shearmodul_voigt(elastic_matrix):
    return (
        (elastic_matrix[0, 0] + elastic_matrix[1, 1] + elastic_matrix[2, 2])
        - (elastic_matrix[0, 1] + elastic_matrix[0, 2] + elastic_matrix[1, 2])
        + 3 * (elastic_matrix[3, 3] + elastic_matrix[4, 4] + elastic_matrix[5, 5])
    ) / 15


def get_youngsmodul_voigt(bulkmodul_voigt, shearmodul_voigt):
    return (9 * bulkmodul_voigt * shearmodul_voigt) / (
        3 * bulkmodul_voigt + shearmodul_voigt
    )


def get_poissonsratio_voigt(bulkmodul_voigt, shearmodul_voigt):
    return (1.5 * bulkmodul_voigt - shearmodul_voigt) / (
        3 * bulkmodul_voigt + shearmodul_voigt
    )


def get_bulkmodul_reuss(elastic_matrix_inverse):
    return 1 / (
        elastic_matrix_inverse[0, 0]
        + elastic_matrix_inverse[1, 1]
        + elastic_matrix_inverse[2, 2]
        + 2
        * (
            elastic_matrix_inverse[0, 1]
            + elastic_matrix_inverse[0, 2]
            + elastic_matrix_inverse[1, 2]
        )
    )


def get_shearmodul_reuss(elastic_matrix_inverse):
    return 15 / (
        4
        * (
            elastic_matrix_inverse[0, 0]
            + elastic_matrix_inverse[1, 1]
            + elastic_matrix_inverse[2, 2]
        )
        - 4
        * (
            elastic_matrix_inverse[0, 1]
            + elastic_matrix_inverse[0, 2]
            + elastic_matrix_inverse[1, 2]
        )
        + 3
        * (
            elastic_matrix_inverse[3, 3]
            + elastic_matrix_inverse[4, 4]
            + elastic_matrix_inverse[5, 5]
        )
    )


def get_youngsmodul_reuss(bulkmodul_reuss, shearmodul_reuss):
    return (9 * bulkmodul_reuss * shearmodul_reuss) / (
        3 * bulkmodul_reuss + shearmodul_reuss
    )


def get_poissonsratio_reuss(bulkmodul_reuss, shearmodul_reuss):
    return (1.5 * bulkmodul_reuss - shearmodul_reuss) / (
        3 * bulkmodul_reuss + shearmodul_reuss
    )


def get_bulkmodul_hill(bulkmodul_voigt, bulkmodul_reuss):
    return _hill_approximation(voigt=bulkmodul_voigt, reuss=bulkmodul_reuss)


def get_shearmodul_hill(shearmodul_voigt, shearmodul_reuss):
    return _hill_approximation(voigt=shearmodul_voigt, reuss=shearmodul_reuss)


def get_youngsmodul_hill(bulkmodul_hill, shearmodul_hill):
    return (9.0 * bulkmodul_hill * shearmodul_hill) / (
        3.0 * bulkmodul_hill + shearmodul_hill
    )


def get_poissonsratio_hill(bulkmodul_hill, shearmodul_hill):
    return (1.5 * bulkmodul_hill - shearmodul_hill) / (
        3.0 * bulkmodul_hill + shearmodul_hill
    )


def get_AVR(shearmodul_voigt, shearmodul_reuss):
    return (
        100.0
        * (shearmodul_voigt - shearmodul_reuss)
        / (shearmodul_voigt + shearmodul_reuss)
    )


def get_elastic_matrix_eigval(elastic_matrix):
    return np.linalg.eig(elastic_matrix)


def get_elastic_matrix_inverse(elastic_matrix):
    return np.linalg.inv(elastic_matrix)


def _hill_approximation(voigt, reuss):
    return 0.50 * (voigt + reuss)
