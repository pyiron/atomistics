import os

from ase.build import bulk
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators import evaluate_with_hessian
from atomistics.workflows import (
    optimize_positions_and_volume,
    LangevinWorkflow,
    get_hesse_matrix,
    get_tasks_for_harmonic_approximation,
    analyse_results_for_harmonic_approximation,
)


try:
    from pylammpsmpi import LammpsASELibrary
    from atomistics.calculators import (
        evaluate_with_lammpslib,
        evaluate_with_lammpslib_library_interface,
        get_potential_by_name,
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLangevin(unittest.TestCase):
    def test_langevin(self):
        steps = 10
        structure = bulk("Al", cubic=True).repeat([3, 3, 3])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        task_dict, phonopy_obj = get_tasks_for_harmonic_approximation(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        result_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        _ = analyse_results_for_harmonic_approximation(
            phonopy=phonopy_obj,
            output_dict=result_dict,
        )
        workflow_md = LangevinWorkflow(
            structure=structure,
            temperature=1000.0,
            overheat_fraction=2.0,
            damping_timescale=100.0,
            time_step=1,
        )
        eng_pot_lst, eng_kin_lst = [], []
        for i in range(steps):
            task_dict = workflow_md.generate_structures()
            result_dict = evaluate_with_hessian(
                task_dict=task_dict,
                structure_equilibrium=structure,
                force_constants=get_hesse_matrix(phonopy=phonopy_obj),
                bulk_modulus=0,
                shear_modulus=0,
            )
            eng_pot, eng_kin = workflow_md.analyse_structures(output_dict=result_dict)
            eng_pot_lst.append(eng_pot)
            eng_kin_lst.append(eng_kin)
        self.assertEqual(len(eng_pot_lst), steps)
        self.assertEqual(len(eng_kin_lst), steps)
        self.assertTrue(eng_pot_lst[-1] < 0.001)
        self.assertTrue(eng_pot_lst[-1] > 0.0)
        self.assertTrue(eng_kin_lst[-1] < 32)
        self.assertTrue(eng_kin_lst[-1] > 20)
        self.assertTrue(eng_kin_lst[0] < 32)
        self.assertTrue(eng_kin_lst[0] > 20)
