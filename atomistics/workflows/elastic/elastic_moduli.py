from functools import cache

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


class ElasticProperties:
    def __init__(self, elastic_matrix):
        self._elastic_matrix = elastic_matrix

    def elastic_matrix(self):
        return self._elastic_matrix

    @cache
    def elastic_matrix_inverse(self):
        return get_elastic_matrix_inverse(elastic_matrix=self.elastic_matrix())

    @cache
    def bulkmodul_voigt(self):
        return get_bulkmodul_voigt(elastic_matrix=self.elastic_matrix())

    @cache
    def shearmodul_voigt(self):
        return get_shearmodul_voigt(elastic_matrix=self.elastic_matrix())

    @cache
    def bulkmodul_reuss(self):
        return get_bulkmodul_reuss(elastic_matrix_inverse=self.elastic_matrix_inverse())

    @cache
    def shearmodul_reuss(self):
        return get_shearmodul_reuss(
            elastic_matrix_inverse=self.elastic_matrix_inverse()
        )

    @cache
    def bulkmodul_hill(self):
        return get_bulkmodul_hill(
            bulkmodul_voigt=self.bulkmodul_voigt(),
            bulkmodul_reuss=self.bulkmodul_reuss(),
        )

    @cache
    def shearmodul_hill(self):
        return get_shearmodul_hill(
            shearmodul_voigt=self.shearmodul_voigt(),
            shearmodul_reuss=self.shearmodul_reuss(),
        )

    @cache
    def youngsmodul_voigt(self):
        return get_youngsmodul_voigt(
            bulkmodul_voigt=self.bulkmodul_voigt(),
            shearmodul_voigt=self.shearmodul_voigt(),
        )

    @cache
    def poissonsratio_voigt(self):
        return get_poissonsratio_voigt(
            bulkmodul_voigt=self.bulkmodul_voigt(),
            shearmodul_voigt=self.shearmodul_voigt(),
        )

    @cache
    def youngsmodul_reuss(self):
        return get_youngsmodul_reuss(
            bulkmodul_reuss=self.bulkmodul_reuss(),
            shearmodul_reuss=self.shearmodul_reuss(),
        )

    @cache
    def poissonsratio_reuss(self):
        return get_poissonsratio_reuss(
            bulkmodul_reuss=self.bulkmodul_reuss(),
            shearmodul_reuss=self.shearmodul_reuss(),
        )

    @cache
    def youngsmodul_hill(self):
        return get_youngsmodul_hill(
            bulkmodul_hill=self.bulkmodul_hill(), shearmodul_hill=self.shearmodul_hill()
        )

    @cache
    def poissonsratio_hill(self):
        return get_poissonsratio_hill(
            bulkmodul_hill=self.bulkmodul_hill(), shearmodul_hill=self.shearmodul_hill()
        )

    @cache
    def AVR(self):
        return get_AVR(
            shearmodul_voigt=self.shearmodul_voigt(),
            shearmodul_reuss=self.shearmodul_reuss(),
        )

    @cache
    def elastic_matrix_eigval(self):
        return get_elastic_matrix_eigval(elastic_matrix=self.elastic_matrix())
