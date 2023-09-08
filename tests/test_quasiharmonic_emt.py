from ase.build import bulk
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators.emt_ase.calculator import evaluate_with_emt
from atomistics.workflows.quasiharmonic.workflow import QuasiHarmonicWorkflow


class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        calculator = QuasiHarmonicWorkflow(
            structure=bulk("Al", a=4.0, cubic=True),
            num_points=11,
            vol_range=0.05,
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_emt(task_dict=structure_dict)
        eng_internal_dict, mesh_collect_dict, dos_collect_dict = calculator.analyse_structures(output_dict=result_dict)
        tp_collect_dict = calculator.get_thermal_properties(t_min=1, t_max=1500, t_step=50, temperatures=None)
        self.assertEqual(len(eng_internal_dict.keys()), 11)
        self.assertEqual(len(tp_collect_dict.keys()), 11)
