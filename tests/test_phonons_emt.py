from ase.build import bulk
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators.emt_ase.calculator import evaluate_with_emt
from atomistics.workflows.phonons.workflow import PhonopyWorkflow


class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        calculator = PhonopyWorkflow(
            structure=bulk("Al", a=4.0, cubic=True),
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_emt(task_dict=structure_dict)
        mesh_dict, dos_dict = calculator.analyse_structures(output_dict=result_dict)
        self.assertEqual((324, 324), calculator.get_hesse_matrix().shape)
        self.assertTrue('qpoints' in mesh_dict.keys())
        self.assertTrue('weights' in mesh_dict.keys())
        self.assertTrue('frequencies' in mesh_dict.keys())
        self.assertTrue('eigenvectors' in mesh_dict.keys())
        self.assertTrue('group_velocities' in mesh_dict.keys())
        self.assertTrue('frequency_points' in dos_dict.keys())
        self.assertTrue('total_dos' in dos_dict.keys())
