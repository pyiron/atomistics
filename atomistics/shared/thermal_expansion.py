import numpy as np

from atomistics.shared.output import OutputThermalExpansion


class ThermalExpansionProperties:
    def __init__(self, temperatures_lst: np.ndarray, volumes_lst: np.ndarray):
        self._temperatures_lst = temperatures_lst
        self._volumes_lst = volumes_lst

    def volumes(self) -> np.ndarray:
        return self._volumes_lst

    def temperatures(self) -> np.ndarray:
        return self._temperatures_lst


def get_thermal_expansion_output(
    temperatures_lst: np.ndarray, volumes_lst: np.ndarray, output_keys: tuple[str]
) -> dict:
    thermal = ThermalExpansionProperties(
        temperatures_lst=temperatures_lst, volumes_lst=volumes_lst
    )
    return OutputThermalExpansion(
        **{k: getattr(thermal, k) for k in OutputThermalExpansion.keys()}
    ).get(output_keys=output_keys)
