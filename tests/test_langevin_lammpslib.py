import os

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows import LangevinWorkflow


try:
    from atomistics.calculators import (
        evaluate_with_lammpslib_library_interface,
        get_potential_by_name,
        LammpsASELibrary,
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLangevin(unittest.TestCase):
    def test_langevin(self):
        steps = 300
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        workflow = LangevinWorkflow(
            structure=structure,
            temperature=1000.0,
            overheat_fraction=2.0,
            damping_timescale=100.0,
            time_step=1,
        )
        lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=None,
            log_file=None,
            library=None,
            diable_log_file=True,
        )
        eng_pot_lst, eng_kin_lst = [], []
        for i in range(steps):
            task_dict = workflow.generate_structures()
            result_dict = evaluate_with_lammpslib_library_interface(
                task_dict=task_dict,
                potential_dataframe=df_pot_selected,
                lmp=lmp,
            )
            eng_pot, eng_kin = workflow.analyse_structures(output_dict=result_dict)
            eng_pot_lst.append(eng_pot)
            eng_kin_lst.append(eng_kin)
        lmp.close()
        eng_tot_lst = np.array(eng_pot_lst) + np.array(eng_kin_lst)
        eng_tot_mean = np.mean(eng_tot_lst[200:])
        self.assertTrue(-105 < eng_tot_mean)
        self.assertTrue(eng_tot_mean < -103)
