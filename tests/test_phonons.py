import unittest
import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from phonopy.units import VaspToTHz

from atomistics.phonons.calculator import PhonopyCalculator


def get_forces_from_emt(structure):
    structure.calc = EMT()
    return structure.get_forces()


class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        calculator = PhonopyCalculator(
            structure=bulk("Al", a=4.0, cubic=True),
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        structure_dict = calculator.generate_structures()
        force_dict = {
            k: get_forces_from_emt(structure=v)
            for k, v in structure_dict.items()
        }
        mesh_dict, dos_dict = calculator.analyse_structures(output_dict=force_dict)
        self.assertEqual((324, 324), calculator.get_hesse_matrix().shape)
        self.assertTrue('qpoints' in mesh_dict.keys())
        self.assertTrue('weights' in mesh_dict.keys())
        self.assertTrue('frequencies' in mesh_dict.keys())
        self.assertTrue('eigenvectors' in mesh_dict.keys())
        self.assertTrue('group_velocities' in mesh_dict.keys())
        self.assertTrue('frequency_points' in dos_dict.keys())
        self.assertTrue('total_dos' in dos_dict.keys())
