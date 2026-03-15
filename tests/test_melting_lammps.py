import os

import unittest


try:
    from atomistics.calculators import get_potential_by_name
    from atomistics.calculators.lammps.melting import estimate_melting_temperature

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLammpsMelting(unittest.TestCase):
    def test_estimate_melting_temperature(self):
        potential = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        melting_temp = estimate_melting_temperature(
            element="Al", 
            potential=potential, 
            strain_run_time_steps=1000, 
            temperature_left=0, 
            temperature_right=1000, 
            number_of_atoms=8000, 
            seed=None,
        )
        self.assertIn(melting_temp, [1008, 1023])
