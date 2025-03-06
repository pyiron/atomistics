import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter, UnitCellFilter
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import optimize_positions, optimize_volume, optimize_positions_and_volume


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

    def test_optimize_volume_unitcellfilter(self):
        structure = bulk("Al", a=4.0, cubic=True)
        task_dict = optimize_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=BFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
            filter_class=UnitCellFilter,
        )
        structure_optimized = result_dict["structure_with_optimized_volume"]
        self.assertAlmostEqual(structure_optimized.get_volume(), 63.72555643511074)

    def test_optimize_volume_frechetcellfilter(self):
        structure = bulk("Al", a=4.0, cubic=True)
        task_dict = optimize_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=BFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
            filter_class=FrechetCellFilter,
        )
        structure_optimized = result_dict["structure_with_optimized_volume"]
        self.assertAlmostEqual(structure_optimized.get_volume(), 63.72555416398717)

    def test_optimize_positions_and_volume_unitcellfilter(self):
        structure = bulk("Al", a=4.0, cubic=True)
        positions_before_displacement = structure.positions.copy()
        structure.positions[0] += [0.01, 0.01, 0.01]
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=BFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
            filter_class=UnitCellFilter,
        )
        structure_optimized = result_dict["structure_with_optimized_positions_and_volume"]
        strain = (structure.get_volume() / structure_optimized.get_volume()) ** (1/3)
        self.assertTrue(
            all(
                np.isclose(
                    positions_before_displacement,
                    (structure_optimized.positions - structure_optimized.positions[0]) * strain,
                    atol=0.000001,
                ).flatten()
            )
        )
        self.assertAlmostEqual(structure_optimized.get_volume(), 63.725564942401235)

    def test_optimize_positions_and_volume_frechetcellfilter(self):
        structure = bulk("Al", a=4.0, cubic=True)
        positions_before_displacement = structure.positions.copy()
        structure.positions[0] += [0.01, 0.01, 0.01]
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=BFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
            filter_class=FrechetCellFilter,
        )
        structure_optimized = result_dict["structure_with_optimized_positions_and_volume"]
        strain = (structure.get_volume() / structure_optimized.get_volume()) ** (1/3)
        self.assertTrue(
            all(
                np.isclose(
                    positions_before_displacement,
                    (structure_optimized.positions - structure_optimized.positions[0]) * strain,
                    atol=0.000001,
                ).flatten()
            )
        )
        self.assertAlmostEqual(structure_optimized.get_volume(), 63.725565063797)
