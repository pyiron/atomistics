import os

from ase.build import bulk
from phonopy.units import VaspToTHz
import unittest

from atomistics.workflows import PhonopyWorkflow, optimize_positions_and_volume

try:
    from atomistics.calculators import (
        evaluate_with_lammps, get_potential_by_name
    )

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
            potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1',
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_lammps(
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
        result_dict = evaluate_with_lammps(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        mesh_dict, dos_dict = workflow.analyse_structures(output_dict=result_dict)
        self.assertEqual((324, 324), workflow.get_hesse_matrix().shape)
        self.assertTrue('qpoints' in mesh_dict.keys())
        self.assertTrue('weights' in mesh_dict.keys())
        self.assertTrue('frequencies' in mesh_dict.keys())
        self.assertTrue('eigenvectors' in mesh_dict.keys())
        self.assertTrue('group_velocities' in mesh_dict.keys())
        self.assertTrue('frequency_points' in dos_dict.keys())
        self.assertTrue('total_dos' in dos_dict.keys())
