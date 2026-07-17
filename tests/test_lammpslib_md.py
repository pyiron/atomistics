import inspect
import os

from ase.build import bulk
import unittest


try:
    from jinja2 import Template
    from pylammpsmpi import LammpsASELibrary
    from lammpsparser import get_potential_by_name
    from atomistics.calculators import (
        calc_molecular_dynamics_nvt_with_lammpslib,
        calc_molecular_dynamics_npt_with_lammpslib,
        calc_molecular_dynamics_nph_with_lammpslib,
        calc_molecular_dynamics_langevin_with_lammpslib,
    )
    from atomistics.calculators.lammps.helpers import lammps_get_structure

    from atomistics.calculators.lammps.commands import (
        LAMMPS_ENSEMBLE_NVT,
        LAMMPS_RUN,
        LAMMPS_THERMO,
        LAMMPS_THERMO_STYLE,
        LAMMPS_TIMESTEP,
        LAMMPS_VELOCITY,
    )
    from atomistics.calculators.lammps.helpers import (
        lammps_calc_md,
        lammps_get_structure,
        lammps_run,
    )

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

    def test_lammps_md_nvt_all_lib(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=None,
            log_file=None,
            library=None,
            disable_log_file=True,
        )
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_VELOCITY,
                LAMMPS_ENSEMBLE_NVT,
            ]
        )
        input_template = Template(init_str).render(
            thermo=10,
            Tstart=100,
            temp=100,
            Tstop=100,
            Tdamp=0.1,
            timestep=0.001,
            seed=4928459,
            dist="gaussian",
            velocity_rescale_factor=2.0,
        )
        run_str = LAMMPS_RUN + "\n"
        lmp_instance = lammps_run(
            structure=structure,
            potential_dataframe=df_pot_selected,
            input_template=input_template,
            lmp=lmp,
        )
        result_dict = lammps_calc_md(
            lmp_instance=lmp_instance,
            run_str=run_str,
            run=100,
            thermo=10,
            output_keys=["positions", "velocities", "cell", "forces", "temperature", "energy_pot", "energy_tot", "pressure"],
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
        structure_md = lammps_get_structure(
            structure=structure,
            lmp_instance=lmp,
            set_velocities=True,
            scale_atoms=True,
            set_cell=True,
        )
        self.assertEqual(structure.get_volume(), structure_md.get_volume())
        self.assertTrue(sum(structure_md.get_velocities() ** 2) > 0)
        lmp.close()

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
            velocity_rescale_factor=None,
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

    def test_lammps_md_npt_all_couple_xyz(self):
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
            couple_xyz=True,
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
            velocity_rescale_factor=None,
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
