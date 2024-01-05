from unittest import TestCase
from atomistics.shared.output import (
    OutputStatic,
    OutputMolecularDynamics,
    OutputThermalExpansion,
    OutputThermodynamic,
    OutputEnergyVolumeCurve,
    OutputElastic,
    OutputPhonons,
)


class TestSharedOutput(TestCase):

    def test_return_none(self):
        for output_cls in [
            OutputStatic,
            OutputMolecularDynamics,
            OutputThermalExpansion,
            OutputThermodynamic,
            OutputEnergyVolumeCurve,
            OutputElastic,
            OutputPhonons,
        ]:
            output_cls.__abstractmethods__ = set()

            class Demo(output_cls):
                pass

            dm = Demo()

            for func in dm.get_keys():
                self.assertIsNone(getattr(dm, func)())
