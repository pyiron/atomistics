import os

from ase import units
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
import numpy as np
import unittest

from atomistics.workflows import calc_molecular_dynamics_thermal_expansion
from atomistics.calculators import calc_molecular_dynamics_thermal_expansion_with_ase


try:
    from atomistics.calculators import (
        calc_molecular_dynamics_thermal_expansion_with_lammpslib,
        evaluate_with_lammpslib,
        get_potential_by_name,
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestMolecularDynamicsThermalExpansion(unittest.TestCase):
    def test_calc_thermal_expansion_using_evaluate(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        task_dict = calc_molecular_dynamics_thermal_expansion(structure=structure)
        result_dict = evaluate_with_lammpslib(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
            lmp_optimizer_kwargs={
                "Tstart": 50,
                "Tstop": 500,
                "Tstep": 50,
            },
        )
        temperature_lst = result_dict["volume_over_temperature"][0]
        volume_lst = result_dict["volume_over_temperature"][1]
        self.assertTrue(all(np.array(temperature_lst) < 600))
        self.assertTrue(all(np.array(temperature_lst) > 0))
        self.assertTrue(volume_lst[0] < volume_lst[-1])

    def test_calc_thermal_expansion_using_calc(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        results_dict = calc_molecular_dynamics_thermal_expansion_with_lammpslib(
            structure=structure,
            potential_dataframe=df_pot_selected,
            Tstart=50,
            Tstop=500,
            Tstep=50,
            Tdamp=0.1,
            run=100,
            thermo=100,
            timestep=0.001,
            Pstart=0.0,
            Pstop=0.0,
            Pdamp=1.0,
            seed=4928459,
            dist="gaussian",
            lmp=None,
        )
        self.assertTrue(all(np.array(results_dict["temperatures"]) < 600))
        self.assertTrue(all(np.array(results_dict["temperatures"]) > 0))
        self.assertTrue(results_dict["volumes"][0] < results_dict["volumes"][-1])

    def test_calc_thermal_expansion_using_ase(self):
        structure = bulk("Al", cubic=True)
        cmds = ["pair_style morse/smooth/linear 9.0", "pair_coeff * * 0.5 1.8 2.95"]
        results_dict = calc_molecular_dynamics_thermal_expansion_with_ase(
            structure=structure.copy(),
            ase_calculator=LAMMPSlib(
                lmpcmds=cmds,
                atom_types={"Al": 1},
                keep_alive=True,
            ),
            temperature_start=50,
            temperature_stop=500,
            temperature_step=50,
            run=100,
            thermo=100,
            timestep=1 * units.fs,
            ttime=100 * units.fs,
            pfactor=2e6 * units.GPa * (units.fs**2),
            externalstress=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar,
        )
        self.assertTrue(all(np.array(results_dict["temperatures"]) < 600))
        self.assertTrue(all(np.array(results_dict["temperatures"]) > 0))
        self.assertTrue(results_dict["volumes"][0] < results_dict["volumes"][-1])
