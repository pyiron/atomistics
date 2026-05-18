import os

import unittest
from ase.build import bulk
import numpy as np


try:
    from atomistics.calculators import get_potential_by_name
    from atomistics.calculators.lammps.melting import estimate_melting_temperature, _generate_structure_with_fixed_number_of_atoms

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
        self.assertIn(melting_temp, [992, 1008, 1023, 1039])

    def test_generate_structure_with_fixed_number_of_atoms(self):
        structure_lst = [bulk("Al"), bulk("Al", cubic=True), bulk("Fe"), bulk("Fe", cubic=True), bulk("Mg"), bulk("Mg", orthorhombic=True), bulk("Si"), bulk("Si", orthorhombic=True)]
        number_lst = [10, 100, 100]
        new_lst = []
        for s in structure_lst:
            for n in number_lst:
                new_lst.append(_generate_structure_with_fixed_number_of_atoms
(structure=s, number_of_atoms=n))
        result_lst = [
            125, 125, 125, 
            500, 500, 500, 
            125, 125, 125, 
            250, 250, 250, 
            250, 250, 250, 
            500, 500, 500,
            250, 250, 250,
            500, 500, 500
        ]
        self.assertEqual(result_lst, [len(s) for s in new_lst])
