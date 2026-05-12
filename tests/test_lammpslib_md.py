import inspect
import os
from unittest.mock import patch

from ase.build import bulk
import pandas
import unittest

try:
    from atomistics.calculators import (
        calc_molecular_dynamics_nvt_with_lammpslib,
        calc_molecular_dynamics_npt_with_lammpslib,
        calc_molecular_dynamics_nph_with_lammpslib,
        calc_molecular_dynamics_langevin_with_lammpslib,
        get_potential_by_name,
    )
    from atomistics.calculators.lammps import libcalculator as lammps_libcalculator

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLammpsMD(unittest.TestCase):
    def test_lammps_md_nvt_all(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = calc_molecular_dynamics_nvt_with_lammpslib(
            structure=structure,
            potential_dataframe=df_pot_selected,
            Tstart=100,
            Tstop=100,
            Tdamp=0.1,
            run=100,
            thermo=10,
            timestep=0.001,
            seed=4928459,
            dist="gaussian",
            lmp=None,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertEqual(result_dict["energy_pot"].shape, (10,))
        self.assertEqual(result_dict["energy_tot"].shape, (10,))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 90)
        self.assertTrue(result_dict["temperature"][-1] < 110)

    def test_lammps_md_nvt_all_no_velocity(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = calc_molecular_dynamics_nvt_with_lammpslib(
            structure=structure,
            potential_dataframe=df_pot_selected,
            Tstart=100,
            Tstop=100,
            Tdamp=0.1,
            run=100,
            thermo=10,
            timestep=0.001,
            seed=4928459,
            dist="gaussian",
            lmp=None,
            disable_initial_velocity=True,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertEqual(result_dict["energy_pot"].shape, (10,))
        self.assertEqual(result_dict["energy_tot"].shape, (10,))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 0)
        self.assertTrue(result_dict["temperature"][-1] < 1)

    def test_lammps_md_nvt_select(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = calc_molecular_dynamics_nvt_with_lammpslib(
            structure=structure,
            potential_dataframe=df_pot_selected,
            Tstart=100,
            Tstop=100,
            Tdamp=0.1,
            run=100,
            thermo=10,
            timestep=0.001,
            seed=4928459,
            dist="gaussian",
            lmp=None,
            output_keys=("temperature",),
        )
        self.assertEqual(len(result_dict.keys()), 1)
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertTrue(result_dict["temperature"][-1] > 90)
        self.assertTrue(result_dict["temperature"][-1] < 110)

    def test_lammps_md_npt_all(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = calc_molecular_dynamics_npt_with_lammpslib(
            structure=structure,
            potential_dataframe=df_pot_selected,
            Tstart=100,
            Tstop=100,
            Tdamp=0.1,
            Pstart=0.0,
            Pstop=0.0,
            Pdamp=1.0,
            run=100,
            thermo=10,
            timestep=0.001,
            seed=4928459,
            dist="gaussian",
            lmp=None,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertEqual(result_dict["energy_pot"].shape, (10,))
        self.assertEqual(result_dict["energy_tot"].shape, (10,))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 90)
        self.assertTrue(result_dict["temperature"][-1] < 110)

    def test_lammps_md_npt_all_no_velocity(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = calc_molecular_dynamics_npt_with_lammpslib(
            structure=structure,
            potential_dataframe=df_pot_selected,
            Tstart=100,
            Tstop=100,
            Tdamp=0.1,
            Pstart=0.0,
            Pstop=0.0,
            Pdamp=1.0,
            run=100,
            thermo=10,
            timestep=0.001,
            seed=4928459,
            dist="gaussian",
            lmp=None,
            disable_initial_velocity=True,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertEqual(result_dict["energy_pot"].shape, (10,))
        self.assertEqual(result_dict["energy_tot"].shape, (10,))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 0)
        self.assertTrue(result_dict["temperature"][-1] < 1)

    def test_lammps_md_nph_all(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = calc_molecular_dynamics_nph_with_lammpslib(
            structure=structure,
            potential_dataframe=df_pot_selected,
            Tstart=100,
            Pstart=0.0,
            Pstop=0.0,
            Pdamp=1.0,
            run=100,
            thermo=10,
            timestep=0.001,
            seed=4928459,
            dist="gaussian",
            lmp=None,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertEqual(result_dict["energy_pot"].shape, (10,))
        self.assertEqual(result_dict["energy_tot"].shape, (10,))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 90)
        self.assertTrue(result_dict["temperature"][-1] < 110)

    def test_lammps_md_langevin_all(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = calc_molecular_dynamics_langevin_with_lammpslib(
            structure=structure,
            potential_dataframe=df_pot_selected,
            Tstart=100,
            Tstop=100,
            Tdamp=0.1,
            run=100,
            thermo=10,
            timestep=0.001,
            seed=4928459,
            dist="gaussian",
            lmp=None,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertEqual(result_dict["energy_pot"].shape, (10,))
        self.assertEqual(result_dict["energy_tot"].shape, (10,))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 90)
        self.assertTrue(result_dict["temperature"][-1] < 130)

    def test_calc_molecular_dynamics_signature(self):
        self.assertEqual(
            inspect.signature(calc_molecular_dynamics_nvt_with_lammpslib)
            .parameters["output_keys"]
            .default,
            (
                "positions",
                "cell",
                "forces",
                "temperature",
                "energy_pot",
                "energy_tot",
                "pressure",
                "velocities",
                "volume",
            ),
        )
        self.assertFalse(
            inspect.signature(calc_molecular_dynamics_npt_with_lammpslib)
            .parameters["couple_xyz"]
            .default
        )

    def test_lammps_md_npt_couple_xyz_command(self):
        with (
            patch.object(
                lammps_libcalculator, "lammps_run", return_value=object()
            ) as lammps_run_mock,
            patch.object(
                lammps_libcalculator, "lammps_calc_md", return_value={}
            ) as lammps_calc_md_mock,
            patch.object(lammps_libcalculator, "lammps_shutdown") as shutdown_mock,
        ):
            calc_molecular_dynamics_npt_with_lammpslib(
                structure=bulk("Al", cubic=True),
                potential_dataframe=pandas.DataFrame(),
                disable_initial_velocity=True,
                couple_xyz=True,
            )
        self.assertIn("couple xyz", lammps_run_mock.call_args.kwargs["input_template"])
        lammps_calc_md_mock.assert_called_once()
        shutdown_mock.assert_called_once()

    def test_lammps_md_npt_without_couple_xyz_command(self):
        with (
            patch.object(
                lammps_libcalculator, "lammps_run", return_value=object()
            ) as lammps_run_mock,
            patch.object(lammps_libcalculator, "lammps_calc_md", return_value={}),
            patch.object(lammps_libcalculator, "lammps_shutdown"),
        ):
            calc_molecular_dynamics_npt_with_lammpslib(
                structure=bulk("Al", cubic=True),
                potential_dataframe=pandas.DataFrame(),
                disable_initial_velocity=True,
                couple_xyz=False,
            )
        self.assertNotIn(
            "couple xyz", lammps_run_mock.call_args.kwargs["input_template"]
        )
