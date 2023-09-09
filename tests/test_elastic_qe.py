import shutil

from ase.build import bulk
import unittest

from atomistics.calculators.quantumespresso_ase.calculator import evaluate_with_quantumespresso
from atomistics.workflows.elastic.workflow import ElasticMatrixWorkflow


cp2k_command = "pw.x"
if shutil.which(cp2k_command) is not None:
    skip_cp2k_test = False
else:
    skip_cp2k_test = True


def validate_elastic_constants(elastic_matrix):
    lst = [
        elastic_matrix[0, 0] > 42,
        elastic_matrix[1, 1] > 42,
        elastic_matrix[2, 2] > 42,
        elastic_matrix[0, 0] < 43,
        elastic_matrix[1, 1] < 43,
        elastic_matrix[2, 2] < 43,
        elastic_matrix[0, 1] > 68,
        elastic_matrix[0, 2] > 68,
        elastic_matrix[1, 2] > 68,
        elastic_matrix[0, 1] < 69,
        elastic_matrix[0, 2] < 69,
        elastic_matrix[1, 2] < 69,
        elastic_matrix[3, 3] > 51,
        elastic_matrix[4, 4] > 51,
        elastic_matrix[5, 5] > 51,
        elastic_matrix[3, 3] < 53,
        elastic_matrix[4, 4] < 53,
        elastic_matrix[5, 5] < 53,
    ]
    if not all(lst):
        print(elastic_matrix)
    return lst


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
        self.assertTrue(all(validate_elastic_constants(elastic_matrix=elastic_dict["C"])))
