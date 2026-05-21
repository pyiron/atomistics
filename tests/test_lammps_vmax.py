import unittest

try:
    from atomistics.calculators.lammps.libcalculator import _get_vmax_command

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestGetVmaxCommand(unittest.TestCase):
    def test_pressure_float_vmax_none(self):
        result = _get_vmax_command(pressure=0.0, vmax=None)
        self.assertEqual(result, "fix ensemble all box/relax iso 0.0")

    def test_pressure_float_vmax_float(self):
        result = _get_vmax_command(pressure=0.0, vmax=0.1)
        self.assertEqual(result, "fix ensemble all box/relax iso 0.0 vmax 0.1")

    def test_pressure_len_3(self):
        result = _get_vmax_command(pressure=[1.0, 2.0, 3.0], vmax=None)
        self.assertEqual(result, "fix ensemble all box/relax x 1.0 y 2.0 z 3.0")

    def test_pressure_len_3_with_vmax(self):
        result = _get_vmax_command(pressure=[1.0, 2.0, 3.0], vmax=0.1)
        self.assertEqual(result, "fix ensemble all box/relax x 1.0 y 2.0 z 3.0 vmax 0.1")

    def test_pressure_len_3_with_none(self):
        result = _get_vmax_command(pressure=[1.0, None, 3.0], vmax=None)
        self.assertEqual(result, "fix ensemble all box/relax x 1.0 z 3.0")

    def test_pressure_len_6(self):
        result = _get_vmax_command(
            pressure=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vmax=None
        )
        self.assertEqual(
            result, "fix ensemble all box/relax x 1.0 y 2.0 z 3.0 xy 4.0 xz 5.0 yz 6.0"
        )

    def test_pressure_len_6_with_vmax(self):
        result = _get_vmax_command(
            pressure=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vmax=0.1
        )
        self.assertEqual(
            result,
            "fix ensemble all box/relax x 1.0 y 2.0 z 3.0 xy 4.0 xz 5.0 yz 6.0 vmax 0.1",
        )

    def test_pressure_invalid_len_raises_value_error(self):
        with self.assertRaises(ValueError):
            _get_vmax_command(pressure=[1.0, 2.0], vmax=None)

    def test_pressure_additional_invalid_lens_raise_value_error(self):
        invalid_pressures = [[], [1.0, 2.0, 3.0, 4.0], [1.0] * 5, [1.0] * 7]
        for pressure in invalid_pressures:
            with self.subTest(pressure=pressure):
                with self.assertRaises(ValueError):
                    _get_vmax_command(pressure=pressure, vmax=None)

    def test_vmax_integer_raises_type_error(self):
        with self.assertRaises(TypeError):
            _get_vmax_command(pressure=0.0, vmax=1)

    def test_vmax_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            _get_vmax_command(pressure=0.0, vmax="0.1")
