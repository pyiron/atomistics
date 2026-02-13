import os
import unittest
from ase.build import bulk

try:
    from atomistics.calculators import get_potential_by_name
    from atomistics.calculators.lammps.melting import estimate_melting_temperature_using_bisection_CNA

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True

@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLammpsMelting(unittest.TestCase):
    def test_estimate_melting_temperature(self):
        input_structure = bulk("Al", cubic=True)
        potential_dataframe = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        melting_temp = estimate_melting_temperature_using_bisection_CNA(
            structure=input_structure,
            potential_dataframe=potential_dataframe,
            target_number_of_atoms=4000,
            temperature_left=0,
            temperature_right=1000,
            temperature_diff_tolerance=10,
            run=1000,
            seed=None,
            cores=1,
            log_file=None,
        )
        self.assertIn(melting_temp, [1008, 1023])
