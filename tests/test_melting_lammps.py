import os

import unittest


try:
    from atomistics.calculators import get_potential_by_name
    from atomistics.calculators.lammps.melting import estimate_melting_temperature_using_bisection_CNA
    from ase.build import bulk

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLammpsMelting(unittest.TestCase):
    def test_estimate_melting_temperature(self):
        potential_dataframe = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        structure = bulk("Al", cubic=True)
        melting_temp = estimate_melting_temperature_using_bisection_CNA(
            structure=structure,
            potential_dataframe=potential_dataframe,
            target_number_of_atoms=4000,
            run=1000,
            temperature_left=0,
            temperature_right=1000,
            seed=None,
        )
        self.assertIn(melting_temp, [992, 1008, 1023, 1039])
