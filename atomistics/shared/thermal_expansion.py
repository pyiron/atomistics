from atomistics.shared.output import OutputThermalExpansion


class ThermalExpansionOutputWrapper(OutputThermalExpansion):
    def __init__(self, temperatures_lst, volumes_lst):
        self._temperatures_lst = temperatures_lst
        self._volumes_lst = volumes_lst

    @property
    def volumes(self):
        return self._volumes_lst

    @property
    def temperatures(self):
        return self._temperatures_lst


def get_thermal_expansion_output(temperatures_lst, volumes_lst, output_keys):
    return ThermalExpansionOutputWrapper(
        temperatures_lst=temperatures_lst, volumes_lst=volumes_lst
    ).get_output(output_keys=output_keys)
