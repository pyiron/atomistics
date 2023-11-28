import os

from ase.build import bulk
import unittest

from atomistics.workflows import calc_molecular_dynamics_thermal_expansion


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
class TestMolecularDynamicsThermalExpansion(unittest.TestCase):
    def test_calc_thermal_expansion(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1',
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        task_dict = calc_molecular_dynamics_thermal_expansion(structure=structure)
        result_dict = evaluate_with_lammps(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
            lmp_optimizer_kwargs={
                "Tstart": 50,
                "Tstop": 500,
                "Tstep": 50,
            }
        )
        temperature_lst = result_dict['volume_over_temperature'][0]
        volume_lst = result_dict['volume_over_temperature'][0]
        self.assertEqual(temperature_lst, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        self.assertTrue(volume_lst[0] < volume_lst[-1])