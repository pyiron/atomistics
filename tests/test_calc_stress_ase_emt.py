from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import optimize_volume


class TestStess(unittest.TestCase):
    def test_calc_stress(self):
        structure = bulk("Al", cubic=True)
        task_dict = {"calc_stress": structure}
        stress_pre_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
        )
        task_dict = optimize_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
        )
        task_dict = {
            "calc_stress": result_dict["structure_with_optimized_volume"]
        }
        stress_post_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
        )
        stress_bool_mat = stress_pre_dict["stress"] > stress_post_dict["stress"]
        self.assertTrue(stress_bool_mat[0][0])
        self.assertTrue(stress_bool_mat[1][1])
        self.assertTrue(stress_bool_mat[2][2])
