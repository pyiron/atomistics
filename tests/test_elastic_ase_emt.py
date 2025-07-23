from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import analyse_results_for_elastic_matrix, get_tasks_for_elastic_matrix, optimize_volume


class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        structure = bulk("Al", cubic=True)
        task_dict = optimize_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
        )
        sym_dict, structure_dict = get_tasks_for_elastic_matrix(
            structure=result_dict["structure_with_optimized_volume"],
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
        )
        result_dict = evaluate_with_ase(task_dict={"calc_energy": structure_dict}, ase_calculator=EMT())
        sym_dict, elastic_dict = analyse_results_for_elastic_matrix(
            output_dict=result_dict,
            sym_dict=sym_dict,
            fit_order=2,
        )
        self.assertAlmostEqual(elastic_dict["elastic_matrix"][0, 0], 53.33436324)
        self.assertAlmostEqual(elastic_dict["elastic_matrix"][0, 1], 32.85853415)
        self.assertAlmostEqual(elastic_dict["elastic_matrix"][3, 3], 36.19530571)
