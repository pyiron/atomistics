import unittest

from phonopy.units import VaspToTHz

from atomistics.workflows.phonons.workflow import PhonopyWorkflow

from .available_evaluators import AL_UNIT, Calculators, EVALUATION_FUNCTIONS


class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        workflow = PhonopyWorkflow(
            structure=AL_UNIT,
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        structure_dict = workflow.generate_structures()

        for calculator in [Calculators.emt, Calculators.gpaw, Calculators.lammps]:
            evaluate = EVALUATION_FUNCTIONS[calculator]
            if evaluate is not None:
                with self.subTest(f"Evaluating with {calculator}"):
                    result_dict = evaluate(structure_dict)
                    mesh_dict, dos_dict = workflow.analyse_structures(
                        output_dict=result_dict
                    )
                    self.assertEqual((324, 324), workflow.get_hesse_matrix().shape)
                    self.assertTrue('qpoints' in mesh_dict.keys())
                    self.assertTrue('weights' in mesh_dict.keys())
                    self.assertTrue('frequencies' in mesh_dict.keys())
                    self.assertTrue('eigenvectors' in mesh_dict.keys())
                    self.assertTrue('group_velocities' in mesh_dict.keys())
                    self.assertTrue('frequency_points' in dos_dict.keys())
                    self.assertTrue('total_dos' in dos_dict.keys())
