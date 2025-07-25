import os
import shutil
import subprocess

from ase.build import bulk
from phonopy.units import VaspToTHz
import unittest

from atomistics.workflows import (
    analyse_results_for_harmonic_approximation,
    get_band_structure,
    get_dynamical_matrix,
    get_hesse_matrix,
    get_tasks_for_harmonic_approximation,
    get_thermal_properties_for_harmonic_approximation,
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
class TestPhonons(unittest.TestCase):
    def setUp(self):
        self.working_directory = os.path.abspath(os.path.join(__file__, "..", "lammps"))
        os.makedirs(self.working_directory, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)

    def test_calc_phonons(self):
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
        task_dict, phonopy_obj = get_tasks_for_harmonic_approximation(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        result_dict = evaluate_with_lammpsfile(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
            working_directory=self.working_directory,
            executable_function=evaluate_lammps,
        )
        phonopy_dict = analyse_results_for_harmonic_approximation(
            output_dict=result_dict,
            phonopy=phonopy_obj,
            dos_mesh=20,
        )
        mesh_dict, dos_dict = phonopy_dict["mesh_dict"], phonopy_dict["total_dos_dict"]
        self.assertEqual((324, 324), get_hesse_matrix(phonopy=phonopy_obj).shape)
        self.assertTrue("qpoints" in mesh_dict.keys())
        self.assertTrue("weights" in mesh_dict.keys())
        self.assertTrue("frequencies" in mesh_dict.keys())
        self.assertTrue("eigenvectors" in mesh_dict.keys())
        self.assertTrue("group_velocities" in mesh_dict.keys())
        self.assertTrue("frequency_points" in dos_dict.keys())
        self.assertTrue("total_dos" in dos_dict.keys())
        thermal_dict = get_thermal_properties_for_harmonic_approximation(
            phonopy=phonopy_obj,
            t_min=1,
            t_max=1500,
            t_step=50,
            temperatures=None,
            cutoff_frequency=None,
            pretend_real=False,
            band_indices=None,
            is_projection=False,
        )
        for key in [
            "temperatures",
            "free_energy",
            "volumes",
            "entropy",
            "heat_capacity",
        ]:
            self.assertTrue(len(thermal_dict[key]), 31)
        self.assertEqual(thermal_dict["temperatures"][0], 1.0)
        self.assertEqual(thermal_dict["temperatures"][-1], 1501.0)
        self.assertTrue(thermal_dict["free_energy"][0] < 0.2)
        self.assertTrue(thermal_dict["free_energy"][0] > 0.1)
        self.assertTrue(thermal_dict["free_energy"][-1] < -2.6)
        self.assertTrue(thermal_dict["free_energy"][-1] > -2.7)
        self.assertTrue(thermal_dict["entropy"][0] < 0.1)
        self.assertTrue(thermal_dict["entropy"][0] > 0.0)
        self.assertTrue(thermal_dict["entropy"][-1] < 271)
        self.assertTrue(thermal_dict["entropy"][-1] > 270)
        self.assertTrue(thermal_dict["heat_capacity"][0] < 0.1)
        self.assertTrue(thermal_dict["heat_capacity"][0] > 0.0)
        self.assertTrue(thermal_dict["heat_capacity"][-1] < 100)
        self.assertTrue(thermal_dict["heat_capacity"][-1] > 99)
        self.assertTrue(thermal_dict["volumes"][-1] < 66.5)
        self.assertTrue(thermal_dict["volumes"][-1] > 66.4)
        self.assertTrue(thermal_dict["volumes"][0] < 66.5)
        self.assertTrue(thermal_dict["volumes"][0] > 66.4)
        thermal_dict = get_thermal_properties_for_harmonic_approximation(
            phonopy=phonopy_obj,
            t_min=1,
            t_max=1500,
            t_step=50,
            temperatures=None,
            cutoff_frequency=None,
            pretend_real=False,
            band_indices=None,
            is_projection=False,
            output_keys=["temperatures", "free_energy"],
        )
        self.assertEqual(len(thermal_dict.keys()), 2)
        self.assertEqual(thermal_dict["temperatures"][0], 1.0)
        self.assertEqual(thermal_dict["temperatures"][-1], 1501.0)
        self.assertTrue(thermal_dict["free_energy"][0] < 0.2)
        self.assertTrue(thermal_dict["free_energy"][0] > 0.1)
        self.assertTrue(thermal_dict["free_energy"][-1] < -2.6)
        self.assertTrue(thermal_dict["free_energy"][-1] > -2.7)
        dynmat_shape = get_dynamical_matrix(phonopy=phonopy_obj, npoints=101).shape
        self.assertEqual(dynmat_shape[0], 12)
        self.assertEqual(dynmat_shape[1], 12)
        hessmat_shape = get_hesse_matrix(phonopy=phonopy_obj).shape
        self.assertEqual(hessmat_shape[0], 324)
        self.assertEqual(hessmat_shape[1], 324)
        band_dict = get_band_structure(phonopy=phonopy_obj, npoints=101)
        self.assertEqual(len(band_dict['qpoints']), 6)
        for vec in band_dict['qpoints']:
            self.assertTrue(vec.shape[0] in [34, 39, 78, 101, 95])
            self.assertEqual(vec.shape[1], 3)
        self.assertEqual(len(band_dict['distances']), 6)
        for vec in band_dict['distances']:
            self.assertTrue(vec.shape[0] in [34, 39, 78, 101, 95])
        self.assertEqual(len(band_dict['frequencies']), 6)
        for vec in band_dict['frequencies']:
            self.assertTrue(vec.shape[0] in [34, 39, 78, 101, 95])
            self.assertEqual(vec.shape[1], 12)
        self.assertIsNone(band_dict['eigenvectors'])
        self.assertIsNone(band_dict['group_velocities'])