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
        self._elastic_matrix_inverse = None
        self._bulkmodul_voigt = None
        self._shearmodul_voigt = None
        self._bulkmodul_reuss = None
        self._shearmodul_reuss = None
        self._bulkmodul_hill = None
        self._shearmodul_hill = None

    def get_elastic_matrix(self):
        return self._elastic_matrix

    def get_elastic_matrix_inverse(self):
        if self._elastic_matrix_inverse is None:
            self._elastic_matrix_inverse = get_elastic_matrix_inverse(
                elastic_matrix=self._elastic_matrix
            )
        return self._elastic_matrix_inverse

    def get_bulkmodul_voigt(self):
        if self._bulkmodul_voigt is None:
            self._bulkmodul_voigt = get_bulkmodul_voigt(
                elastic_matrix=self._elastic_matrix
            )
        return self._bulkmodul_voigt

    def get_shearmodul_voigt(self):
        if self._shearmodul_voigt is None:
            self._shearmodul_voigt = get_shearmodul_voigt(
                elastic_matrix=self._elastic_matrix
            )
        return self._shearmodul_voigt

    def get_bulkmodul_reuss(self):
        if self._bulkmodul_reuss is None:
            self._bulkmodul_reuss = get_bulkmodul_reuss(
                elastic_matrix_inverse=self.get_elastic_matrix_inverse()
            )
        return self._bulkmodul_reuss

    def get_shearmodul_reuss(self):
        if self._shearmodul_reuss is None:
            self._shearmodul_reuss = get_shearmodul_reuss(
                elastic_matrix_inverse=self.get_elastic_matrix_inverse()
            )
        return self._shearmodul_reuss

    def get_bulkmodul_hill(self):
        if self._bulkmodul_hill is None:
            self._bulkmodul_hill = get_bulkmodul_hill(
                bulkmodul_voigt=self.get_bulkmodul_voigt(),
                bulkmodul_reuss=self.get_bulkmodul_reuss(),
            )
        return self._bulkmodul_hill

    def get_shearmodul_hill(self):
        if self._shearmodul_hill is None:
            self._shearmodul_hill = get_shearmodul_hill(
                shearmodul_voigt=self.get_shearmodul_voigt(),
                shearmodul_reuss=self.get_shearmodul_reuss(),
            )
        return self._shearmodul_hill

    def get_youngsmodul_voigt(self):
        return get_youngsmodul_voigt(
            bulkmodul_voigt=self.get_bulkmodul_voigt(),
            shearmodul_voigt=self.get_shearmodul_voigt(),
        )

    def get_poissonsratio_voigt(self):
        return get_poissonsratio_voigt(
            bulkmodul_voigt=self.get_bulkmodul_voigt(),
            shearmodul_voigt=self.get_shearmodul_voigt(),
        )

    def get_youngsmodul_reuss(self):
        return get_youngsmodul_reuss(
            bulkmodul_reuss=self.get_bulkmodul_reuss(),
            shearmodul_reuss=self.get_shearmodul_reuss(),
        )

    def get_poissonsratio_reuss(self):
        return get_poissonsratio_reuss(
            bulkmodul_reuss=self.get_bulkmodul_reuss(),
            shearmodul_reuss=self.get_shearmodul_reuss(),
        )

    def get_youngsmodul_hill(self):
        return get_youngsmodul_hill(
            bulkmodul_hill=self.get_bulkmodul_hill(),
            shearmodul_hill=self.get_shearmodul_hill(),
        )

    def get_poissonratio_hill(self):
        return get_poissonsratio_hill(
            bulkmodul_hill=self.get_bulkmodul_hill(),
            shearmodul_hill=self.get_shearmodul_hill(),
        )

    def get_AVR(self):
        return get_AVR(
            shearmodul_voigt=self.get_shearmodul_voigt(),
            shearmodul_reuss=self.get_shearmodul_reuss(),
        )

    def get_elastic_matrix_eigval(self):
        return get_elastic_matrix_eigval(elastic_matrix=self._elastic_matrix)
