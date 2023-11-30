from ase.build import bulk
import numpy as np
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import ElasticMatrixWorkflow

try:
    from gpaw import GPAW, PW

    skip_gpaw_test = False
except ImportError:
    skip_gpaw_test = True


@unittest.skipIf(
    skip_gpaw_test, "gpaw is not installed, so the gpaw tests are skipped."
)
class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        workflow = ElasticMatrixWorkflow(
            structure=bulk("Al", a=4.0, cubic=True),
            num_of_point=5,
            eps_range=0.05,
            sqrt_eta=True,
            fit_order=2
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=GPAW(
                xc="PBE",
                mode=PW(300),
                kpts=(3, 3, 3)
            )
        )
        elastic_dict = workflow.analyse_structures(output_dict=result_dict)
        self.assertTrue(np.isclose(elastic_dict["C"][0, 0], 125.66807354, atol=1e-04))
        self.assertTrue(np.isclose(elastic_dict["C"][0, 1], 68.41418321, atol=1e-04))
        self.assertTrue(np.isclose(elastic_dict["C"][3, 3], 99.29916329, atol=1e-04))
