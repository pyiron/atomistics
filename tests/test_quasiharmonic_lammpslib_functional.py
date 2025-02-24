import os

from ase.build import bulk
import numpy as np
from phonopy.units import VaspToTHz
import unittest

from atomistics.workflows.quasiharmonic import (
    generate_structures_helper,
    analyse_structures_helper,
    get_thermal_properties,
)

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
        result_dict = evaluate_with_lammpslib(
            task_dict={"optimize_positions_and_volume": structure},
            potential_dataframe=df_pot_selected,
        )
        phonopy_dict, repeat_vector, structure_energy_dict, structure_forces_dict = (
            generate_structures_helper(
                structure=result_dict["structure_with_optimized_positions_and_volume"],
                vol_range=0.05,
                num_points=11,
                strain_lst=None,
                displacement=0.01,
                number_of_snapshots=None,
                interaction_range=10,
                factor=VaspToTHz,
            )
        )
        result_dict = evaluate_with_lammpslib(
            task_dict={
                "calc_energy": structure_energy_dict,
                "calc_forces": structure_forces_dict,
            },
            potential_dataframe=df_pot_selected,
        )
        eng_internal_dict, phonopy_collect_dict = analyse_structures_helper(
            phonopy_dict=phonopy_dict,
            output_dict=result_dict,
            dos_mesh=20,
            number_of_snapshots=None,
        )
        tp_collect_dict = get_thermal_properties(
            eng_internal_dict=eng_internal_dict,
            phonopy_dict=phonopy_dict,
            structure_dict=structure_energy_dict,
            repeat_vector=repeat_vector,
            fit_type="polynomial",
            fit_order=3,
            t_min=1,
            t_max=1500,
            t_step=50,
            temperatures=None,
        )
        for key in [
            "temperatures",
            "free_energy",
            "volumes",
            "entropy",
            "heat_capacity",
        ]:
            self.assertTrue(len(tp_collect_dict[key]), 31)
        self.assertEqual(tp_collect_dict["temperatures"][0], 1.0)
        self.assertEqual(tp_collect_dict["temperatures"][-1], 1501.0)
        self.assertTrue(tp_collect_dict["free_energy"][0] < 0.2)
        self.assertTrue(tp_collect_dict["free_energy"][0] > 0.1)
        self.assertTrue(tp_collect_dict["free_energy"][-1] < -2.6)
        self.assertTrue(tp_collect_dict["free_energy"][-1] > -2.7)
        self.assertTrue(tp_collect_dict["entropy"][0] < 0.1)
        self.assertTrue(tp_collect_dict["entropy"][0] > 0.0)
        self.assertTrue(tp_collect_dict["entropy"][-1] < 273)
        self.assertTrue(tp_collect_dict["entropy"][-1] > 272)
        self.assertTrue(tp_collect_dict["heat_capacity"][0] < 0.1)
        self.assertTrue(tp_collect_dict["heat_capacity"][0] > 0.0)
        self.assertTrue(tp_collect_dict["heat_capacity"][-1] < 100)
        self.assertTrue(tp_collect_dict["heat_capacity"][-1] > 99)
        self.assertTrue(tp_collect_dict["volumes"][-1] < 68.6)
        self.assertTrue(tp_collect_dict["volumes"][-1] > 68.5)
        self.assertTrue(tp_collect_dict["volumes"][0] < 66.8)
        self.assertTrue(tp_collect_dict["volumes"][0] > 66.7)
        thermal_properties_dict = get_thermal_properties(
            eng_internal_dict=eng_internal_dict,
            phonopy_dict=phonopy_dict,
            structure_dict=structure_energy_dict,
            repeat_vector=repeat_vector,
            fit_type="polynomial",
            fit_order=3,
            temperatures=[100, 1000],
            output_keys=["free_energy", "free_energy", "temperatures", "volumes"],
            quantum_mechanical=True,
        )
        temperatures_qh_qm, volumes_qh_qm = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        thermal_properties_dict = get_thermal_properties(
            eng_internal_dict=eng_internal_dict,
            phonopy_dict=phonopy_dict,
            structure_dict=structure_energy_dict,
            repeat_vector=repeat_vector,
            fit_type="polynomial",
            fit_order=3,
            temperatures=[100, 1000],
            output_keys=["free_energy", "temperatures", "volumes"],
            quantum_mechanical=False,
        )
        temperatures_qh_cl, volumes_qh_cl = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        self.assertEqual(len(eng_internal_dict.keys()), 11)
        self.assertEqual(len(tp_collect_dict.keys()), 5)
        self.assertEqual(len(temperatures_qh_qm), 2)
        self.assertEqual(len(volumes_qh_qm), 2)
        self.assertTrue(volumes_qh_qm[0] < volumes_qh_qm[-1])
        self.assertEqual(len(temperatures_qh_cl), 2)
        self.assertEqual(len(volumes_qh_cl), 2)
        self.assertTrue(volumes_qh_cl[0] < volumes_qh_cl[-1])
