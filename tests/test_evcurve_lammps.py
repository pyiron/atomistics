import os

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows import EnergyVolumeCurveWorkflow, optimize_positions_and_volume
from atomistics.workflows.evcurve.workflow import (
    analyse_structures_helper,
    generate_structures_helper,
    get_thermal_properties,
)


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
        thermal_properties_dict = workflow.get_thermal_properties(
            temperatures=[100, 1000],
            output_keys=["temperatures", "volumes"]
        )
        temperatures_ev, volumes_ev = thermal_properties_dict["temperatures"], thermal_properties_dict["volumes"]
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 66.43019853103964))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 77.7250135953191))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 1.2795467367276832))
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])

    def test_calc_evcurve_functional(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1',
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = evaluate_with_lammps(
            task_dict={"optimize_positions_and_volume": structure},
            potential_dataframe=df_pot_selected,
        )
        structure_dict = generate_structures_helper(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            vol_range=0.05,
            num_points=11,
            strain_lst=None,
            axes=('x', 'y', 'z'),
        )
        result_dict = evaluate_with_lammps(
            task_dict={"calc_energy": structure_dict},
            potential_dataframe=df_pot_selected,
        )
        fit_dict = analyse_structures_helper(
            output_dict=result_dict,
            structure_dict=structure_dict,
            fit_type="polynomial",
            fit_order=3
        )
        thermal_properties_dict = get_thermal_properties(
            fit_dict=fit_dict,
            masses=structure.get_masses(),
            t_min=1.0,
            t_max=1500.0,
            t_step=50.0,
            temperatures=[100, 1000],
            constant_volume=False,
            output_keys=["temperatures", "volumes"],
        )
        temperatures_ev, volumes_ev = thermal_properties_dict["temperatures"], thermal_properties_dict["volumes"]
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 66.43019853103964))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 77.7250135953191))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 1.2795467367276832))
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])
