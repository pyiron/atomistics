from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import ElasticMatrixWorkflow


class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        workflow = ElasticMatrixWorkflow(
            structure=bulk("Al", a=4.0, cubic=True),
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
            fit_order=2
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(task_dict=task_dict, ase_calculator=EMT())
        elastic_dict = workflow.analyse_structures(output_dict=result_dict)
        self.assertTrue(np.isclose(elastic_dict["C"][0, 0], 52.62435421))
        self.assertTrue(np.isclose(elastic_dict["C"][0, 1], 32.6743838))
        self.assertTrue(np.isclose(elastic_dict["C"][3, 3], 35.58677436))
