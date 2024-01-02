from functools import cached_property

import numpy as np

from atomistics.shared.output import Output


def get_bulkmodul_voigt(elastic_matrix) -> float:
    return (
        elastic_matrix[0, 0]
        + elastic_matrix[1, 1]
        + elastic_matrix[2, 2]
        + 2 * (elastic_matrix[0, 1] + elastic_matrix[0, 2] + elastic_matrix[1, 2])
    ) / 9


def get_shearmodul_voigt(elastic_matrix) -> float:
    return (
        (elastic_matrix[0, 0] + elastic_matrix[1, 1] + elastic_matrix[2, 2])
        - (elastic_matrix[0, 1] + elastic_matrix[0, 2] + elastic_matrix[1, 2])
        + 3 * (elastic_matrix[3, 3] + elastic_matrix[4, 4] + elastic_matrix[5, 5])
    ) / 15


def get_youngsmodul_voigt(bulkmodul_voigt, shearmodul_voigt) -> float:
    return (9 * bulkmodul_voigt * shearmodul_voigt) / (
        3 * bulkmodul_voigt + shearmodul_voigt
    )


def get_poissonsratio_voigt(bulkmodul_voigt, shearmodul_voigt) -> float:
    return (1.5 * bulkmodul_voigt - shearmodul_voigt) / (
        3 * bulkmodul_voigt + shearmodul_voigt
    )


def get_bulkmodul_reuss(elastic_matrix_inverse) -> float:
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


def get_shearmodul_reuss(elastic_matrix_inverse) -> float:
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


def get_youngsmodul_reuss(bulkmodul_reuss, shearmodul_reuss) -> float:
    return (9 * bulkmodul_reuss * shearmodul_reuss) / (
        3 * bulkmodul_reuss + shearmodul_reuss
    )


def get_poissonsratio_reuss(bulkmodul_reuss, shearmodul_reuss) -> float:
    return (1.5 * bulkmodul_reuss - shearmodul_reuss) / (
        3 * bulkmodul_reuss + shearmodul_reuss
    )


def get_bulkmodul_hill(bulkmodul_voigt, bulkmodul_reuss) -> float:
    return _hill_approximation(voigt=bulkmodul_voigt, reuss=bulkmodul_reuss)


def get_shearmodul_hill(shearmodul_voigt, shearmodul_reuss) -> float:
    return _hill_approximation(voigt=shearmodul_voigt, reuss=shearmodul_reuss)


def get_youngsmodul_hill(bulkmodul_hill, shearmodul_hill) -> float:
    return (9.0 * bulkmodul_hill * shearmodul_hill) / (
        3.0 * bulkmodul_hill + shearmodul_hill
    )


def get_poissonsratio_hill(bulkmodul_hill, shearmodul_hill) -> float:
    return (1.5 * bulkmodul_hill - shearmodul_hill) / (
        3.0 * bulkmodul_hill + shearmodul_hill
    )


def get_AVR(shearmodul_voigt, shearmodul_reuss) -> float:
    return (
        100.0
        * (shearmodul_voigt - shearmodul_reuss)
        / (shearmodul_voigt + shearmodul_reuss)
    )


def get_elastic_matrix_eigval(elastic_matrix) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eig(elastic_matrix)


def get_elastic_matrix_inverse(elastic_matrix) -> np.ndarray:
    return np.linalg.inv(elastic_matrix)


def _hill_approximation(voigt, reuss):
    return 0.50 * (voigt + reuss)


class OutputElastic(Output):
    def __init__(self, elastic_matrix: np.ndarray):
        self._elastic_matrix = elastic_matrix

    @property
    def elastic_matrix(self):
        return self._elastic_matrix

    @cached_property
    def _elastic_matrix_inverse(self):
        return get_elastic_matrix_inverse(elastic_matrix=self.elastic_matrix)

    @cached_property
    def bulkmodul_voigt(self):
        return get_bulkmodul_voigt(elastic_matrix=self.elastic_matrix)

    @cached_property
    def shearmodul_voigt(self):
        return get_shearmodul_voigt(elastic_matrix=self.elastic_matrix)

    @cached_property
    def bulkmodul_reuss(self):
        return get_bulkmodul_reuss(elastic_matrix_inverse=self._elastic_matrix_inverse)

    @cached_property
    def shearmodul_reuss(self):
        return get_shearmodul_reuss(elastic_matrix_inverse=self._elastic_matrix_inverse)

    @cached_property
    def bulkmodul_hill(self):
        return get_bulkmodul_hill(
                bulkmodul_voigt=self.bulkmodul_voigt,
                bulkmodul_reuss=self.bulkmodul_reuss,
            )

    @cached_property
    def shearmodul_hill(self):
        return get_shearmodul_hill(
                shearmodul_voigt=self.shearmodul_voigt,
                shearmodul_reuss=self.shearmodul_reuss,
            )

    @cached_property
    def youngsmodul_voigt(self):
        return get_youngsmodul_voigt(
            bulkmodul_voigt=self.bulkmodul_voigt,
            shearmodul_voigt=self.shearmodul_voigt,
        )

    @cached_property
    def poissonsratio_voigt(self):
        return get_poissonsratio_voigt(
            bulkmodul_voigt=self.bulkmodul_voigt,
            shearmodul_voigt=self.shearmodul_voigt,
        )

    @cached_property
    def youngsmodul_reuss(self):
        return get_youngsmodul_reuss(
            bulkmodul_reuss=self.bulkmodul_reuss,
            shearmodul_reuss=self.shearmodul_reuss,
        )

    @cached_property
    def poissonsratio_reuss(self):
        return get_poissonsratio_reuss(
            bulkmodul_reuss=self.bulkmodul_reuss,
            shearmodul_reuss=self.shearmodul_reuss,
        )

    @cached_property
    def youngsmodul_hill(self):
        return get_youngsmodul_hill(
            bulkmodul_hill=self.bulkmodul_hill,
            shearmodul_hill=self.shearmodul_hill,
        )

    @cached_property
    def poissonsratio_hill(self):
        return get_poissonsratio_hill(
            bulkmodul_hill=self.bulkmodul_hill,
            shearmodul_hill=self.shearmodul_hill,
        )

    @cached_property
    def AVR(self):
        return get_AVR(
            shearmodul_voigt=self.shearmodul_voigt,
            shearmodul_reuss=self.shearmodul_reuss,
        )

    @cached_property
    def elastic_matrix_eigval(self):
        return get_elastic_matrix_eigval(elastic_matrix=self.elastic_matrix)

    def get(self, *output: str) -> dict:
        return {q: getattr(self, q) for q in output}

    @classmethod
    def fields(cls):
        return tuple(
            q for q in dir(cls)
            if not (q[0] == "_" or q in ["get", "fields"])
        )
