import os
import shutil
import subprocess

from ase.build import bulk
import unittest

from atomistics.workflows import (
    EnergyVolumeCurveWorkflow,
    optimize_positions_and_volume,
)

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


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def setUp(self):
        self.working_directory = os.path.abspath(os.path.join(__file__, "..", "lammps"))
        os.makedirs(self.working_directory, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)

    def test_calc_evcurve(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_lammpsfile(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
            working_directory=self.working_directory,
            executable_function=evaluate_lammps,
        )
        workflow = EnergyVolumeCurveWorkflow(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            num_points=11,
            fit_type="polynomial",
            fit_order=3,
            vol_range=0.05,
            axes=("x", "y", "z"),
            strains=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_lammpsfile(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
            working_directory=self.working_directory,
            executable_function=evaluate_lammps,
        )
        fit_dict = workflow.analyse_structures(output_dict=result_dict)
        thermal_properties_dict = workflow.get_thermal_properties(
            temperatures=[100, 1000], output_keys=["temperatures", "volumes"]
        )
        temperatures_ev, volumes_ev = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        self.assertAlmostEqual(fit_dict["volume_eq"], 66.43019790724603)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 77.72501703646152)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 1.2795467367276832)
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])
