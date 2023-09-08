import os
import shutil

from ase.build import bulk
import numpy as np
import unittest

from atomistics.calculators.cp2k_ase.calculator import evaluate_with_cp2k
from atomistics.workflows.elastic.workflow import ElasticMatrixWorkflow


cp2k_command = "cp2k_shell.sopt"
if shutil.which(cp2k_command) is not None:
    skip_cp2k_test = False
else:
    skip_cp2k_test = True


@unittest.skipIf(
    skip_cp2k_test, "cp2k is not installed, so the cp2k tests are skipped."
)
class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        resource_path = os.path.join(os.path.dirname(__file__), "static", "cp2k")
        calculator = ElasticMatrixWorkflow(
            structure=bulk("Al", a=4.0, cubic=True),
            num_of_point=5,
            eps_range=0.05,
            sqrt_eta=True,
            fit_order=2
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_cp2k(
            task_dict=structure_dict,
            command=cp2k_command,
            basis_set_file=os.path.join(resource_path, 'BASIS_SET'),
            basis_set="DZVP-GTH-PADE",
            potential_file=os.path.join(resource_path, 'GTH_POTENTIALS'),
            pseudo_potential="GTH-PADE-q4",
        )
        elastic_dict = calculator.analyse_structures(output_dict=result_dict)
        print(elastic_dict)
        self.assertTrue(np.isclose(elastic_dict["C"][0, 0], 125.66807354, atol=1e-04))
        self.assertTrue(np.isclose(elastic_dict["C"][0, 1], 68.41418321, atol=1e-04))
        self.assertTrue(np.isclose(elastic_dict["C"][3, 3], 99.29916329, atol=1e-04))
