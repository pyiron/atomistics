import unittest
from atomistics.calculators.lammps.libcalculator import _get_vmax_command
from atomistics.calculators.lammps.commands import LAMMPS_MINIMIZE_VOLUME

class TestLammpsVmax(unittest.TestCase):
    def test_get_vmax_command_none(self):
        self.assertEqual(_get_vmax_command(vmax=None), LAMMPS_MINIMIZE_VOLUME)

    def test_get_vmax_command_float(self):
        vmax = 0.001
        self.assertEqual(
            _get_vmax_command(vmax=vmax),
            LAMMPS_MINIMIZE_VOLUME + f" vmax {vmax}"
        )

    def test_get_vmax_command_error(self):
        with self.assertRaises(TypeError):
            _get_vmax_command(vmax=1)
        with self.assertRaises(TypeError):
            _get_vmax_command(vmax="0.001")
