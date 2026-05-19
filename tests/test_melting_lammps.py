import os
import unittest

try:
    from atomistics.calculators import get_potential_by_name
    from atomistics.calculators.lammps.melting import (
        estimate_melting_temperature_using_bisection_CNA, 
        _generate_structure_with_fixed_number_of_atoms
    )
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
            temperature_left=0,
            temperature_right=1000,
            run=1000,
            optimization_maxiter=10000,
            seed=None,
        )

        self.assertIn(melting_temp, [977, 992, 1008, 1023, 1039, 1055])

    def test_generate_structure_with_fixed_number_of_atoms(self):
        structure_lst = [
            bulk("Al"), 
            bulk("Al", cubic=True), 
            bulk("Fe"), 
            bulk("Fe", cubic=True), 
            bulk("Mg"), 
            bulk("Mg", orthorhombic=True), 
            bulk("Si"), 
            bulk("Si", orthorhombic=True)
        ]

        number_lst = [10, 100, 100]
        new_lst = []
        for s in structure_lst:
            for n in number_lst:
                new_lst.append(
                    _generate_structure_with_fixed_number_of_atoms(structure=s, number_of_atoms=n)
                    )
                
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