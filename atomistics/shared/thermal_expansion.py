from atomistics.shared.output import OutputThermalExpansion


class ThermalExpansionProperties:
    def __init__(self, temperatures_lst, volumes_lst):
        self._temperatures_lst = temperatures_lst
        self._volumes_lst = volumes_lst

    def volumes(self):
        return self._volumes_lst

    def temperatures(self):
        return self._temperatures_lst


def get_thermal_expansion_output(temperatures_lst, volumes_lst, output):
    thermal_properties = ThermalExpansionProperties(
        temperatures_lst=temperatures_lst, volumes_lst=volumes_lst
    )
    return OutputThermalExpansion(
        temperatures=thermal_properties.temperatures,
        volumes=thermal_properties.volumes,
    ).get(output=output)
