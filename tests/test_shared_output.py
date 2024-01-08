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
                def __init__(self):
                    super().__init__()
                    self._demo = None

            dm = Demo()

            for func in dir(dm):
                if func[0] != "_" and func not in ['keys', 'get_output']:
                    self.assertIsNone(getattr(dm, func))
