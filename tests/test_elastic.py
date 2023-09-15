import unittest

from atomistics._tests import AL_UNIT, Calculators, EVALUATION_FUNCTIONS
from atomistics.workflows.elastic.workflow import ElasticMatrixWorkflow


class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        workflow = ElasticMatrixWorkflow(
            structure=AL_UNIT,
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
            fit_order=2
        )
        structure_dict = workflow.generate_structures()

        for calculator, expected in [
            (Calculators.emt, (46.08382141, 30.74709713, 30.22242313)),
            (Calculators.gpaw, (125.66807354, 68.41418321, 99.29916329)),  # WILL FAIL -- data is for a=4.0
            (Calculators.lammps, (114.10393023, 60.51098897, 51.23931149)),
        ]:
            evaluate = EVALUATION_FUNCTIONS[calculator]
            if evaluate is not None:
                with self.subTest(f"Evaluating with {calculator}"):
                    result_dict = evaluate(structure_dict)
                    elastic_dict = workflow.analyse_structures(output_dict=result_dict)
                    print(elastic_dict)
                    self.assertAlmostEqual(elastic_dict["C"][0, 0], expected[0])
                    self.assertAlmostEqual(elastic_dict["C"][0, 1], expected[1])
                    self.assertAlmostEqual(elastic_dict["C"][3, 3], expected[2])
