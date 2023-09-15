import unittest

from phonopy.units import VaspToTHz

from atomistics._tests import AL_UNIT, Calculators, EVALUATION_FUNCTIONS
from atomistics.workflows.quasiharmonic.workflow import QuasiHarmonicWorkflow


class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        workflow = QuasiHarmonicWorkflow(
            structure=AL_UNIT,
            num_points=11,
            vol_range=0.05,
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
                    eng_internal_dict, mesh_collect_dict, dos_collect_dict = \
                        workflow.analyse_structures(output_dict=result_dict)
                    tp_collect_dict = workflow.get_thermal_properties(
                        t_min=1, t_max=1500, t_step=50, temperatures=None
                    )
                    self.assertEqual(len(eng_internal_dict.keys()), 11)
                    self.assertEqual(len(tp_collect_dict.keys()), 11)
