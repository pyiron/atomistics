import os

from ase.build import bulk
import numpy as np
import unittest
import pandas

from atomistics.workflows import EnergyVolumeCurveWorkflow, optimize_positions_and_volume


try:
    import matgl
    from matgl.ext.ase import M3GNetCalculator
    from atomistics.calculators import (
        evaluate_with_lammps, get_potential_by_name
    )

    skip_lammps_matgl_test = False
except ImportError:
    skip_lammps_matgl_test = True


@unittest.skipIf(
    skip_lammps_matgl_test, "LAMMPS and or matgl are not installed, so the corresponding tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = pandas.DataFrame({
            "Config": ["['pair_style m3gnet " + os.path.abspath(os.path.join(__file__, "..", "static", "lammps", "potential_LAMMPS", "M3GNET")) + "\n', 'pair_coeff * * M3GNet-MP-2021.2.8-PES Al\n']"],
            "Filename": [[]],
            "Species": [["Al"]]
        })
        print(df_pot_selected["Config"].values[0])
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
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 66.56048874824006, atol=1e-04))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 50.96266448851179, atol=1e-02))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 4.674534962000779, atol=1e-02))
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])
