import os

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows import EnergyVolumeCurveWorkflow, optimize_positions_and_volume


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
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
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
        workflow = EnergyVolumeCurveWorkflow(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            num_points=11,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=('x', 'y', 'z'),
            strains=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_lammps(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        fit_dict = workflow.analyse_structures(output_dict=result_dict)
        temperatures_ev, volumes_ev = workflow.get_thermal_expansion(output_dict=result_dict, temperatures=[100, 1000])
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 66.43019853103964))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 77.7250135953191))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 1.2795467367276832))
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])
