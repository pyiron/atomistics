import unittest
import numpy as np

from ase.build import bulk
from atomistics.elastic.calculator import ElasticMatrixCalculator

from emt_helper import evaluate_with_emt


class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        calculator = calculator = ElasticMatrixCalculator(
            structure=bulk("Al", a=4.0, cubic=True),
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
            fit_order=2
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_emt(task_dict=structure_dict)
        elastic_dict = calculator.analyse_structures(output_dict=result_dict)
        self.assertTrue(np.isclose(elastic_dict["C"][0, 0], 52.62435421))
        self.assertTrue(np.isclose(elastic_dict["C"][0, 1], 32.6743838))
        self.assertTrue(np.isclose(elastic_dict["C"][3, 3], 35.58677436))
