from ase.build import bulk
import numpy as np
import unittest

from atomistics.calculators.gpaw_ase.calculator import evaluate_with_gpaw
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow


class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        calculator = EnergyVolumeCurveWorkflow(
            structure=bulk("Al", a=4.0, cubic=True),
            num_points=11,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=['x', 'y', 'z'],
            strains=None,
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_gpaw(
            task_dict=structure_dict,
            xc="PBE",
            encut=500,
            kpts=(5, 5, 5)
        )
        fit_dict = calculator.analyse_structures(output_dict=result_dict)
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 63.72615218844302))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 39.544084907317895))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 2.2509394023322566))
