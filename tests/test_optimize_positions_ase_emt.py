import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import optimize_positions


class TestOptimizePositionsEMT(unittest.TestCase):
    def test_optimize_positions(self):
        structure = bulk("Al", a=4.0, cubic=True)
        positions_before_displacement = structure.positions.copy()
        structure.positions[0] += [0.01, 0.01, 0.01]
        task_dict = optimize_positions(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=BFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
        )
        structure_optimized = result_dict["structure_with_optimized_positions"]
        self.assertTrue(
            all(
                np.isclose(
                    positions_before_displacement,
                    structure_optimized.positions - structure_optimized.positions[0],
                ).flatten()
            )
        )
