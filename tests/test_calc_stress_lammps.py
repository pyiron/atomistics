import os

from ase.build import bulk
import unittest

from atomistics.workflows import optimize_positions_and_volume


try:
    from atomistics.calculators import evaluate_with_lammpslib, get_potential_by_name

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestStess(unittest.TestCase):
    def test_calc_stress(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        task_dict = {"calc_stress": structure}
        stress_pre_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        task_dict = {
            "calc_stress": result_dict["structure_with_optimized_positions_and_volume"]
        }
        stress_post_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        stress_bool_mat = stress_pre_dict["stress"] > stress_post_dict["stress"]
        self.assertTrue(stress_bool_mat[0][0])
        self.assertTrue(stress_bool_mat[1][1])
        self.assertTrue(stress_bool_mat[2][2])
