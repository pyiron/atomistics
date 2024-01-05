from atomistics.shared.output import OutputThermalExpansion


class ThermalExpansionProperties:
    def __init__(self, temperatures_lst, volumes_lst):
        self._temperatures_lst = temperatures_lst
        self._volumes_lst = volumes_lst

    def get_volumes(self):
        return self._volumes_lst

    def get_temperatures(self):
        return self._temperatures_lst


def get_thermal_expansion_output(temperatures_lst, volumes_lst, output):
    thermal_properties = ThermalExpansionProperties(
        temperatures_lst=temperatures_lst, volumes_lst=volumes_lst
    )
    return OutputThermalExpansion(
        temperatures=thermal_properties.get_temperatures,
        volumes=thermal_properties.get_volumes,
    ).get(*output)
