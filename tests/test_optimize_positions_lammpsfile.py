import os
import shutil
import subprocess

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows import optimize_positions

try:
    from atomistics.calculators.lammps.potential import get_potential_by_name
    from atomistics.calculators.lammps.filecalculator import evaluate_with_lammpsfile

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


def evaluate_lammps(working_directory):
    command = "mpiexec -n 1 --oversubscribe lmp_mpi -in lmp.in"
    output = subprocess.check_output(
        command, cwd=working_directory, shell=True, universal_newlines=True, env=os.environ.copy()
    )
    print(output)
    return output


def optimize_structure(structure, potential_name, working_directory, resource_path=None):
    df_pot_selected = get_potential_by_name(
        potential_name=potential_name,
        resource_path=resource_path,
    )
    task_dict = optimize_positions(structure=structure)
    result_dict = evaluate_with_lammpsfile(
        task_dict=task_dict,
        potential_dataframe=df_pot_selected,
        working_directory=working_directory,
        executable_function=evaluate_lammps,
        lmp_optimizer_kwargs={"ftol": 0.000001},
    )
    return result_dict["structure_with_optimized_positions"]


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestOptimizePositionsLAMMPS(unittest.TestCase):
    def setUp(self):
        self.working_directory = os.path.abspath(os.path.join(__file__, "..", "lammps"))
        os.makedirs(self.working_directory, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)

    def test_optimize_positions_with_resource_path(self):
        structure = bulk("Al", cubic=True)
        positions_before_displacement = structure.positions.copy()
        structure.positions[0] += [0.01, 0.01, 0.01]
        structure_optimized = optimize_structure(
            structure=structure,
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
            working_directory=self.working_directory,
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
            working_directory=self.working_directory,
        )
        self.assertTrue(
            all(
                np.isclose(
                    positions_before_displacement,
                    structure_optimized.positions - structure_optimized.positions[0],
                ).flatten()
            )
        )
