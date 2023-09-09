import shutil

from ase.build import bulk
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators.quantumespresso_ase.calculator import evaluate_with_quantumespresso
from atomistics.workflows.phonons.workflow import PhonopyWorkflow


cp2k_command = "pw.x"
if shutil.which(cp2k_command) is not None:
    skip_cp2k_test = False
else:
    skip_cp2k_test = True


@unittest.skipIf(
    skip_cp2k_test, "Quantum Espresso is not installed, so the Quantum Espresso tests are skipped."
)
class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        pseudopotentials = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"}
        calculator = PhonopyWorkflow(
            structure=bulk("Al", a=4.15, cubic=True),
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_quantumespresso(
            task_dict=structure_dict,
            pseudopotentials=pseudopotentials,
            tstress=True,
            tprnfor=True,
            kpts=(1, 1, 1),
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
