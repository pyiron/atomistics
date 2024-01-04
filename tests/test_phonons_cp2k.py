import os
import shutil

from ase.build import bulk
from ase.calculators.cp2k import CP2K
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows.phonons.workflow import PhonopyWorkflow


cp2k_command = "cp2k.ssmp"
if shutil.which(cp2k_command) is not None:
    skip_cp2k_test = False
else:
    skip_cp2k_test = True


@unittest.skipIf(
    skip_cp2k_test, "cp2k is not installed, so the cp2k tests are skipped."
)
class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        resource_path = os.path.join(os.path.dirname(__file__), "static", "cp2k")
        calculator = PhonopyWorkflow(
            structure=bulk("Al", a=4.05, cubic=True),
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_ase(
            task_dict=structure_dict,
            ase_calculator=CP2K(
                command=cp2k_command,
                basis_set_file=os.path.join(resource_path, 'BASIS_SET'),
                basis_set="DZVP-GTH-PADE",
                potential_file=os.path.join(resource_path, 'GTH_POTENTIALS'),
                pseudo_potential="GTH-PADE-q4",
            )
        )
        mesh_dict, dos_dict = calculator.analyse_structures(output_dict=result_dict)
        self.assertEqual((324, 324), calculator.get_hesse_matrix().shape)
        self.assertTrue('qpoints' in mesh_dict.keys())
        self.assertTrue('weights' in mesh_dict.keys())
        self.assertTrue('frequencies' in mesh_dict.keys())
        self.assertTrue('eigenvectors' in mesh_dict.keys())
        self.assertTrue('group_velocities' in mesh_dict.keys())
        self.assertTrue('frequency_points' in dos_dict.keys())
        self.assertTrue('total_dos' in dos_dict.keys())
