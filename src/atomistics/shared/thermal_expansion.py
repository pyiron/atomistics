import numpy as np

from atomistics.shared.output import OutputThermalExpansion


class ThermalExpansionProperties:
    def __init__(self, temperatures_lst: np.ndarray, volumes_lst: np.ndarray):
        """
        Initialize the ThermalExpansionProperties class.

        Parameters:
        temperatures_lst (np.ndarray): Array of temperatures.
        volumes_lst (np.ndarray): Array of volumes.
        """
        self._temperatures_lst = temperatures_lst
        self._volumes_lst = volumes_lst

    def volumes(self) -> np.ndarray:
        """
        Get the array of volumes.

        Returns:
        np.ndarray: Array of volumes.
        """
        return self._volumes_lst

    def temperatures(self) -> np.ndarray:
        """
        Get the array of temperatures.

        Returns:
        np.ndarray: Array of temperatures.
        """
        return self._temperatures_lst


def get_thermal_expansion_output(
    temperatures_lst: np.ndarray, volumes_lst: np.ndarray, output_keys: tuple[str]
) -> dict:
    """
    Get the thermal expansion output.

    Parameters:
    temperatures_lst (np.ndarray): Array of temperatures.
    volumes_lst (np.ndarray): Array of volumes.
    output_keys (tuple[str]): Tuple of output keys.

    Returns:
    dict: Dictionary containing the thermal expansion output.
    """
    thermal = ThermalExpansionProperties(
        temperatures_lst=temperatures_lst, volumes_lst=volumes_lst
    )
    return OutputThermalExpansion(
        **{k: getattr(thermal, k) for k in OutputThermalExpansion.keys()}
    ).get(output_keys=output_keys)
