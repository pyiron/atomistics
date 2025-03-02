import os

from ase.build import bulk
from phonopy.units import VaspToTHz
import unittest

from atomistics.workflows import PhonopyWorkflow, optimize_positions_and_volume

try:
    from atomistics.calculators import evaluate_with_lammpslib, get_potential_by_name

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        workflow = PhonopyWorkflow(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        phonopy_dict = workflow.analyse_structures(output_dict=result_dict)
        mesh_dict, dos_dict = phonopy_dict["mesh_dict"], phonopy_dict["total_dos_dict"]
        self.assertEqual((324, 324), workflow.get_hesse_matrix().shape)
        self.assertTrue("qpoints" in mesh_dict.keys())
        self.assertTrue("weights" in mesh_dict.keys())
        self.assertTrue("frequencies" in mesh_dict.keys())
        self.assertTrue("eigenvectors" in mesh_dict.keys())
        self.assertTrue("group_velocities" in mesh_dict.keys())
        self.assertTrue("frequency_points" in dos_dict.keys())
        self.assertTrue("total_dos" in dos_dict.keys())
        thermal_dict = workflow.get_thermal_properties(
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
        thermal_dict = workflow.get_thermal_properties(
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
        dynmat_shape = workflow.get_dynamical_matrix().shape
        self.assertEqual(dynmat_shape[0],12)
        self.assertEqual(dynmat_shape[1], 12)
        hessmat_shape = workflow.get_hesse_matrix().shape
        self.assertEqual(hessmat_shape[0], 324)
        self.assertEqual(hessmat_shape[1], 324)
        band_dict = workflow.get_band_structure()
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