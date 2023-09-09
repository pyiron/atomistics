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
    return [
        elastic_matrix[0, 0] > 200,
        elastic_matrix[1, 1] > 200,
        elastic_matrix[2, 2] > 200,
        elastic_matrix[0, 1] > 135,
        elastic_matrix[0, 2] > 135,
        elastic_matrix[1, 2] > 135,
        elastic_matrix[3, 3] > 100,
        elastic_matrix[4, 4] > 100,
        elastic_matrix[5, 5] > 100,
    ]


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
