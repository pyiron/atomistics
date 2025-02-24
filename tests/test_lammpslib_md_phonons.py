import os

from ase.build import bulk
from phonopy.units import VaspToTHz
import unittest

from atomistics.workflows import PhonopyWorkflow


try:
    from atomistics.calculators import (
        calc_molecular_dynamics_phonons_with_lammpslib,
        evaluate_with_lammpslib,
        get_potential_by_name,
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLammpsMD(unittest.TestCase):
    def test_lammps_md_nvt_all(self):
        structure = bulk("Si", cubic=True)
        potential_dataframe = get_potential_by_name(
            potential_name="1988--Tersoff-J--Si-c--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        workflow = PhonopyWorkflow(
            structure=structure,
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=[[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
            number_of_snapshots=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=potential_dataframe,
        )
        workflow.analyse_structures(output_dict=result_dict)
        trajectory = calc_molecular_dynamics_phonons_with_lammpslib(
            structure_ase=structure,
            potential_dataframe=potential_dataframe,
            force_constants=workflow.phonopy.get_force_constants(),
            phonopy_unitcell=workflow.phonopy.get_unitcell(),
            phonopy_primitive_matrix=workflow.phonopy.get_primitive_matrix(),
            phonopy_supercell_matrix=workflow.phonopy.get_supercell_matrix(),
            total_time=20,  # ps
            time_step=0.001,  # ps
            relaxation_time=5,  # ps
            silent=True,
            supercell=[2, 2, 2],
            memmap=False,
            velocity_only=True,
            temperature=600,
        )
        self.assertEqual(trajectory.get_number_of_atoms(), 64)
        self.assertEqual(trajectory.velocity.shape, (20000, 64, 3))
        self.assertEqual(len(trajectory.get_time()), 20000)
