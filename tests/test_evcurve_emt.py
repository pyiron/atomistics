from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import EnergyVolumeCurveWorkflow


class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        workflow = EnergyVolumeCurveWorkflow(
            structure=bulk("Al", a=4.0, cubic=True),
            num_points=11,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=('x', 'y', 'z'),
            strains=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(task_dict=task_dict, ase_calculator=EMT())
        fit_dict = workflow.analyse_structures(output_dict=result_dict)
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 63.72615218844302))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 39.544084907317895))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 2.2509394023322566))
