import shutil

from ase.build import bulk
import numpy as np
import unittest

from atomistics.calculators.quantumespresso_ase.calculator import evaluate_with_quantumespresso
from atomistics.workflows.elastic.workflow import ElasticMatrixWorkflow


cp2k_command = "pw.x"
if shutil.which(cp2k_command) is not None:
    skip_cp2k_test = False
else:
    skip_cp2k_test = True


@unittest.skipIf(
    skip_cp2k_test, "cp2k is not installed, so the cp2k tests are skipped."
)
class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        pseudopotentials = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"}
        calculator = ElasticMatrixWorkflow(
            structure=bulk("Al", a=4.15, cubic=True),
            num_of_point=5,
            eps_range=0.05,
            sqrt_eta=True,
            fit_order=2
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_quantumespresso(
            task_dict=structure_dict,
            pseudopotentials=pseudopotentials,
            tstress=True,
            tprnfor=True,
            kpts=(3, 3, 3),
        )
        elastic_dict = calculator.analyse_structures(output_dict=result_dict)
        print(elastic_dict)
        self.assertTrue(np.isclose(elastic_dict["C"][0, 0], 42.82240443, atol=1e-04))
        self.assertTrue(np.isclose(elastic_dict["C"][0, 1], 68.14834072, atol=1e-04))
        self.assertTrue(np.isclose(elastic_dict["C"][3, 3], 52.27621873, atol=1e-04))
