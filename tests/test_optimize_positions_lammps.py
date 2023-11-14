import os

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows import optimize_positions

try:
    from atomistics.calculators import (
        evaluate_with_lammps, get_potential_dataframe
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestOptimizePositionsLAMMPS(unittest.TestCase):
    def test_optimize_positions(self):
        potential = '1999--Mishin-Y--Al--LAMMPS--ipr1'
        resource_path = os.path.join(os.path.dirname(__file__), "static", "lammps")
        structure = bulk("Al", cubic=True)
        positions_before_displacement = structure.positions.copy()
        structure.positions[0] += [0.01, 0.01, 0.01]
        df_pot = get_potential_dataframe(
            structure=structure,
            resource_path=resource_path
        )
        df_pot_selected = df_pot[df_pot.Name == potential].iloc[0]
        task_dict = optimize_positions(structure=structure)
        result_dict = evaluate_with_lammps(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
            lmp_optimizer_kwargs={"ftol": 0.000001},
        )
        structure_optimized = result_dict["structure_with_optimized_positions"]
        self.assertTrue(
            all(np.isclose(
                positions_before_displacement,
                structure_optimized.positions-structure_optimized.positions[0],
            ).flatten())
        )

