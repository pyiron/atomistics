import os

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows import optimize_positions

try:
    from atomistics.calculators import evaluate_with_lammpslib, get_potential_by_name

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


def optimize_structure(structure, potential_name, resource_path=None):
    df_pot_selected = get_potential_by_name(
        potential_name=potential_name,
        resource_path=resource_path,
    )
    task_dict = optimize_positions(structure=structure)
    result_dict = evaluate_with_lammpslib(
        task_dict=task_dict,
        potential_dataframe=df_pot_selected,
        lmp_optimizer_kwargs={"ftol": 0.000001},
    )
    return result_dict["structure_with_optimized_positions"]


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestOptimizePositionsLAMMPS(unittest.TestCase):
    def test_optimize_positions_with_resource_path(self):
        structure = bulk("Al", cubic=True)
        positions_before_displacement = structure.positions.copy()
        structure.positions[0] += [0.01, 0.01, 0.01]
        structure_optimized = optimize_structure(
            structure=structure,
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        self.assertTrue(
            all(
                np.isclose(
                    positions_before_displacement,
                    structure_optimized.positions - structure_optimized.positions[0],
                ).flatten()
            )
        )

    def test_optimize_positions_without_resource_path(self):
        structure = bulk("Al", cubic=True)
        positions_before_displacement = structure.positions.copy()
        structure.positions[0] += [0.01, 0.01, 0.01]
        structure_optimized = optimize_structure(
            structure=structure,
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=None,
        )
        self.assertTrue(
            all(
                np.isclose(
                    positions_before_displacement,
                    structure_optimized.positions - structure_optimized.positions[0],
                ).flatten()
            )
        )
