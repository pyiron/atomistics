from functools import cache

import numpy as np

from atomistics.shared.output import OutputElastic


def get_bulkmodul_voigt(elastic_matrix: np.ndarray) -> float:
    """
    Calculate the Voigt average of the bulk modulus.

    Parameters:
    elastic_matrix (np.ndarray): The elastic matrix.

    Returns:
    float: The Voigt average of the bulk modulus.
    """
    return (
        elastic_matrix[0, 0]
        + elastic_matrix[1, 1]
        + elastic_matrix[2, 2]
        + 2 * (elastic_matrix[0, 1] + elastic_matrix[0, 2] + elastic_matrix[1, 2])
    ) / 9


def get_shearmodul_voigt(elastic_matrix: np.ndarray) -> float:
    """
    Calculate the Voigt average of the shear modulus.

    Parameters:
    elastic_matrix (np.ndarray): The elastic matrix.

    Returns:
    float: The Voigt average of the shear modulus.
    """
    return (
        (elastic_matrix[0, 0] + elastic_matrix[1, 1] + elastic_matrix[2, 2])
        - (elastic_matrix[0, 1] + elastic_matrix[0, 2] + elastic_matrix[1, 2])
        + 3 * (elastic_matrix[3, 3] + elastic_matrix[4, 4] + elastic_matrix[5, 5])
    ) / 15


def get_youngsmodul_voigt(bulkmodul_voigt: float, shearmodul_voigt: float) -> float:
    """
    Calculates the Young's modulus using the Voigt notation.

    Parameters:
        bulkmodul_voigt (float): The bulk modulus in Voigt notation.
        shearmodul_voigt (float): The shear modulus in Voigt notation.

    Returns:
        float: The Young's modulus calculated using the Voigt notation.
    """
    return (9 * bulkmodul_voigt * shearmodul_voigt) / (
        3 * bulkmodul_voigt + shearmodul_voigt
    )


def get_poissonsratio_voigt(bulkmodul_voigt: float, shearmodul_voigt: float) -> float:
    """
    Calculate the Poisson's ratio using the Voigt notation.

    Parameters:
        bulkmodul_voigt (float): The bulk modulus in Voigt notation.
        shearmodul_voigt (float): The shear modulus in Voigt notation.

    Returns:
        float: The Poisson's ratio calculated using the Voigt notation.
    """
    return (1.5 * bulkmodul_voigt - shearmodul_voigt) / (
        3 * bulkmodul_voigt + shearmodul_voigt
    )


def get_bulkmodul_reuss(elastic_matrix_inverse: np.ndarray) -> float:
    """
    Calculate the Reuss average of the bulk modulus.

    Parameters:
    elastic_matrix_inverse (np.ndarray): The inverse of the elastic matrix.

    Returns:
    float: The Reuss average of the bulk modulus.
    """
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


def get_shearmodul_reuss(elastic_matrix_inverse: np.ndarray) -> float:
    """
    Calculate the Reuss average of the shear modulus.

    Parameters:
    elastic_matrix_inverse (np.ndarray): The inverse of the elastic matrix.

    Returns:
    float: The Reuss average of the shear modulus.
    """
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


def get_youngsmodul_reuss(bulkmodul_reuss: float, shearmodul_reuss: float) -> float:
    """
    Calculates the Young's modulus using the Reuss approximation.

    Parameters:
        bulkmodul_reuss (float): The bulk modulus in Reuss approximation.
        shearmodul_reuss (float): The shear modulus in Reuss approximation.

    Returns:
        float: The calculated Young's modulus.

    """
    return (9 * bulkmodul_reuss * shearmodul_reuss) / (
        3 * bulkmodul_reuss + shearmodul_reuss
    )


def get_poissonsratio_reuss(bulkmodul_reuss: float, shearmodul_reuss: float) -> float:
    """
    Calculate the Poisson's ratio using the Reuss approximation.

    Parameters:
        bulkmodul_reuss (float): The bulk modulus in Reuss approximation.
        shearmodul_reuss (float): The shear modulus in Reuss approximation.

    Returns:
        float: The Poisson's ratio calculated using the Reuss approximation.
    """
    return (1.5 * bulkmodul_reuss - shearmodul_reuss) / (
        3 * bulkmodul_reuss + shearmodul_reuss
    )


def get_bulkmodul_hill(bulkmodul_voigt: float, bulkmodul_reuss: float) -> float:
    """
    Calculate the Hill average of the bulk modulus.

    Parameters:
        bulkmodul_voigt (float): The bulk modulus in Voigt notation.
        bulkmodul_reuss (float): The bulk modulus in Reuss approximation.

    Returns:
        float: The Hill average of the bulk modulus.
    """
    return _hill_approximation(voigt=bulkmodul_voigt, reuss=bulkmodul_reuss)


def get_shearmodul_hill(shearmodul_voigt: float, shearmodul_reuss: float) -> float:
    """
    Calculate the shear modulus using the Hill approximation.

    Args:
        shearmodul_voigt (float): The shear modulus calculated using the Voigt approximation.
        shearmodul_reuss (float): The shear modulus calculated using the Reuss approximation.

    Returns:
        float: The shear modulus calculated using the Hill approximation.
    """
    return _hill_approximation(voigt=shearmodul_voigt, reuss=shearmodul_reuss)


def get_youngsmodul_hill(bulkmodul_hill: float, shearmodul_hill: float) -> float:
    """
    Calculate the Young's modulus using the Hill approximation.

    Parameters:
        bulkmodul_hill (float): The bulk modulus.
        shearmodul_hill (float): The shear modulus.

    Returns:
        float: The calculated Young's modulus.

    """
    return (9.0 * bulkmodul_hill * shearmodul_hill) / (
        3.0 * bulkmodul_hill + shearmodul_hill
    )


def get_poissonsratio_hill(bulkmodul_hill: float, shearmodul_hill: float) -> float:
    """
    Calculate the Poisson's ratio using Hill's approximation.

    Parameters:
    bulkmodul_hill (float): The bulk modulus calculated using Hill's approximation.
    shearmodul_hill (float): The shear modulus calculated using Hill's approximation.

    Returns:
    float: The Poisson's ratio calculated using Hill's approximation.
    """
    return (1.5 * bulkmodul_hill - shearmodul_hill) / (
        3.0 * bulkmodul_hill + shearmodul_hill
    )


def get_AVR(shearmodul_voigt: float, shearmodul_reuss: float) -> float:
    """
    Calculate the average value ratio (AVR) of the shear modulus.

    Parameters:
        shearmodul_voigt (float): The shear modulus calculated using the Voigt approximation.
        shearmodul_reuss (float): The shear modulus calculated using the Reuss approximation.

    Returns:
        float: The average value ratio (AVR) of the shear modulus.
    """
    return (
        100.0
        * (shearmodul_voigt - shearmodul_reuss)
        / (shearmodul_voigt + shearmodul_reuss)
    )


def get_elastic_matrix_eigval(
    elastic_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the eigenvalues and eigenvectors of the elastic matrix.

    Parameters:
        elastic_matrix (np.ndarray): The elastic matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The eigenvalues and eigenvectors of the elastic matrix.
    """
    return np.linalg.eig(elastic_matrix)


def get_elastic_matrix_inverse(elastic_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of the elastic matrix.

    Parameters:
        elastic_matrix (np.ndarray): The elastic matrix.

    Returns:
        np.ndarray: The inverse of the elastic matrix.
    """
    return np.linalg.inv(elastic_matrix)


def _hill_approximation(voigt: float, reuss: float) -> float:
    """
    Calculate the Hill approximation of a property.

    Parameters:
        voigt (float): The property calculated using the Voigt approximation.
        reuss (float): The property calculated using the Reuss approximation.

    Returns:
        float: The Hill approximation of the property.
    """
    return 0.50 * (voigt + reuss)


class ElasticProperties:
    def __init__(self, elastic_matrix: np.ndarray):
        """
        Initialize the ElasticProperties class.

        Parameters:
            elastic_matrix (np.ndarray): The elastic matrix.
        """
        self._elastic_matrix = elastic_matrix

    def elastic_matrix(self) -> np.ndarray:
        """
        Get the elastic matrix.

        Returns:
            np.ndarray: The elastic matrix.
        """
        return self._elastic_matrix

    @cache
    def elastic_matrix_inverse(self) -> np.ndarray:
        """
        Calculate the inverse of the elastic matrix.

        Returns:
            np.ndarray: The inverse of the elastic matrix.
        """
        return get_elastic_matrix_inverse(elastic_matrix=self.elastic_matrix())

    @cache
    def bulkmodul_voigt(self) -> float:
        """
        Calculate the bulk modulus in Voigt notation.

        Returns:
            float: The bulk modulus in Voigt notation.
        """
        return get_bulkmodul_voigt(elastic_matrix=self.elastic_matrix())

    @cache
    def shearmodul_voigt(self) -> float:
        """
        Calculate the shear modulus in Voigt notation.

        Returns:
            float: The shear modulus in Voigt notation.
        """
        return get_shearmodul_voigt(elastic_matrix=self.elastic_matrix())

    @cache
    def bulkmodul_reuss(self) -> float:
        """
        Calculate the Reuss average of the bulk modulus.

        Returns:
            float: The Reuss average of the bulk modulus.
        """
        return get_bulkmodul_reuss(elastic_matrix_inverse=self.elastic_matrix_inverse())

    @cache
    def shearmodul_reuss(self) -> float:
        """
        Calculate the Reuss average of the shear modulus.

        Returns:
            float: The Reuss average of the shear modulus.
        """
        return get_shearmodul_reuss(
            elastic_matrix_inverse=self.elastic_matrix_inverse()
        )

    @cache
    def bulkmodul_hill(self) -> float:
        """
        Calculate the Hill average of the bulk modulus.

        Returns:
            float: The Hill average of the bulk modulus.
        """
        return get_bulkmodul_hill(
            bulkmodul_voigt=self.bulkmodul_voigt(),
            bulkmodul_reuss=self.bulkmodul_reuss(),
        )

    @cache
    def shearmodul_hill(self) -> float:
        """
        Calculate the shear modulus using the Hill approximation.

        Returns:
            float: The shear modulus calculated using the Hill approximation.
        """
        return get_shearmodul_hill(
            shearmodul_voigt=self.shearmodul_voigt(),
            shearmodul_reuss=self.shearmodul_reuss(),
        )

    @cache
    def youngsmodul_voigt(self) -> float:
        """
        Calculate the Young's modulus using the Voigt approximation.

        Returns:
            float: The calculated Young's modulus.
        """
        return get_youngsmodul_voigt(
            bulkmodul_voigt=self.bulkmodul_voigt(),
            shearmodul_voigt=self.shearmodul_voigt(),
        )

    @cache
    def poissonsratio_voigt(self) -> float:
        """
        Calculate the Poisson's ratio using the Voigt approximation.

        Returns:
            float: The Poisson's ratio calculated using the Voigt approximation.
        """
        return get_poissonsratio_voigt(
            bulkmodul_voigt=self.bulkmodul_voigt(),
            shearmodul_voigt=self.shearmodul_voigt(),
        )

    @cache
    def youngsmodul_reuss(self) -> float:
        """
        Calculate the Young's modulus using the Reuss approximation.

        Returns:
            float: The calculated Young's modulus.
        """
        return get_youngsmodul_reuss(
            bulkmodul_reuss=self.bulkmodul_reuss(),
            shearmodul_reuss=self.shearmodul_reuss(),
        )

    @cache
    def poissonsratio_reuss(self) -> float:
        """
        Calculate the Poisson's ratio using the Reuss approximation.

        Returns:
            float: The Poisson's ratio calculated using the Reuss approximation.
        """
        return get_poissonsratio_reuss(
            bulkmodul_reuss=self.bulkmodul_reuss(),
            shearmodul_reuss=self.shearmodul_reuss(),
        )

    @cache
    def youngsmodul_hill(self) -> float:
        """
        Calculate the Young's modulus using the Hill approximation.

        Returns:
            float: The calculated Young's modulus.
        """
        return get_youngsmodul_hill(
            bulkmodul_hill=self.bulkmodul_hill(), shearmodul_hill=self.shearmodul_hill()
        )

    @cache
    def poissonsratio_hill(self) -> float:
        """
        Calculate the Poisson's ratio using Hill's approximation.

        Returns:
            float: The Poisson's ratio calculated using Hill's approximation.
        """
        return get_poissonsratio_hill(
            bulkmodul_hill=self.bulkmodul_hill(), shearmodul_hill=self.shearmodul_hill()
        )

    @cache
    def AVR(self) -> float:
        """
        Calculate the average value ratio (AVR) of the shear modulus.

        Returns:
            float: The average value ratio (AVR) of the shear modulus.
        """
        return get_AVR(
            shearmodul_voigt=self.shearmodul_voigt(),
            shearmodul_reuss=self.shearmodul_reuss(),
        )

    @cache
    def elastic_matrix_eigval(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the eigenvalues and eigenvectors of the elastic matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The eigenvalues and eigenvectors of the elastic matrix.
        """
        return get_elastic_matrix_eigval(elastic_matrix=self.elastic_matrix())

    def to_dict(self, output_keys: tuple = OutputElastic.keys()) -> dict:
        """
        Convert the ElasticProperties object to a dictionary.

        Parameters:
            output_keys (tuple): The keys to include in the output dictionary.

        Returns:
            dict: The ElasticProperties object as a dictionary.
        """
        return OutputElastic(**{k: getattr(self, k) for k in OutputElastic.keys()}).get(
            output_keys=output_keys
        )
