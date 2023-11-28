import unittest

try:
    import pandas
    from atomistics.calculators.lammps.potential import (
        validate_potential_dataframe
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestPotential(unittest.TestCase):
    def test_validate_potential_dataframe(self):
        with self.assertRaises(ValueError):
            _ = validate_potential_dataframe(
                potential_dataframe=pandas.DataFrame({})
            )
        with self.assertRaises(ValueError):
            _ = validate_potential_dataframe(
                potential_dataframe=pandas.DataFrame({"a": [1, 2]})
            )
        with self.assertRaises(TypeError):
            _ = validate_potential_dataframe(
                potential_dataframe=0
            )
        series = validate_potential_dataframe(
            potential_dataframe=pandas.DataFrame({"a": [1]})
        )
        self.assertTrue(isinstance(series, pandas.Series))
