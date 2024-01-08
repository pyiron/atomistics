from atomistics.shared.output import OutputThermalExpansion


class ThermalExpansionProperties:
    def __init__(self, temperatures_lst, volumes_lst):
        self._temperatures_lst = temperatures_lst
        self._volumes_lst = volumes_lst

    def volumes(self):
        return self._volumes_lst

    def temperatures(self):
        return self._temperatures_lst


OutputThermalExpansionProperties = OutputThermalExpansion(
    **{
        k: getattr(ThermalExpansionProperties, k)
        for k in OutputThermalExpansion.keys()
    }
)
