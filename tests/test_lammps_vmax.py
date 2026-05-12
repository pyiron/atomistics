import unittest

try:
    from atomistics.calculators.lammps.commands import LAMMPS_MINIMIZE_VOLUME
    from atomistics.calculators.lammps.libcalculator import _get_vmax_command

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestGetVmaxCommand(unittest.TestCase):
    def test_vmax_none_returns_minimize_volume(self):
        result = _get_vmax_command(vmax=None)
        self.assertEqual(result, LAMMPS_MINIMIZE_VOLUME)

    def test_vmax_float_appends_vmax(self):
        result = _get_vmax_command(vmax=0.1)
        self.assertEqual(result, LAMMPS_MINIMIZE_VOLUME + " vmax 0.1")

    def test_vmax_float_zero_appends_vmax(self):
        result = _get_vmax_command(vmax=0.0)
        self.assertEqual(result, LAMMPS_MINIMIZE_VOLUME + " vmax 0.0")

    def test_vmax_integer_raises_type_error(self):
        with self.assertRaises(TypeError):
            _get_vmax_command(vmax=1)

    def test_vmax_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            _get_vmax_command(vmax="0.1")
