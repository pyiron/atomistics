import os
import unittest

import numpy as np
from ase.build import bulk

try:
    from jinja2 import Template
    from atomistics.calculators import (
        get_energy_pot_average_with_lammpslib,
        get_energy_pot_with_lammpslib,
        get_potential_by_name,
        get_structure_snapshot_with_lammpslib,
        lammps_run,
        lammps_shutdown,
    )
    from atomistics.calculators.lammps.commands import (
        LAMMPS_AVE_ENERGY,
        LAMMPS_ENSEMBLE_NVT,
        LAMMPS_THERMO,
        LAMMPS_THERMO_STYLE,
        LAMMPS_TIMESTEP,
        LAMMPS_VELOCITY,
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


def _nvt_template(thermo=10, timestep=0.001, T=300.0, Tdamp=0.1, seed=4928459, include_velocity=True):
    parts = [LAMMPS_THERMO_STYLE, LAMMPS_TIMESTEP, LAMMPS_THERMO]
    if include_velocity:
        parts.append(LAMMPS_VELOCITY)
    parts.append(LAMMPS_ENSEMBLE_NVT)
    return Template("\n".join(parts)).render(
        thermo=thermo,
        timestep=timestep,
        temp=T,
        Tstart=T,
        Tstop=T,
        Tdamp=Tdamp,
        seed=seed,
        dist="gaussian",
        velocity_rescale_factor=2.0,
    )


def _nvt_with_average_template(window, T=300.0, timestep=0.001, Tdamp=0.1, seed=4928459, include_velocity=True, fix_id="avePE"):
    base = _nvt_template(
        thermo=window, timestep=timestep, T=T, Tdamp=Tdamp, seed=seed, include_velocity=include_velocity
    )
    ave_energy = Template(LAMMPS_AVE_ENERGY).render(
        energy_variable="myPE", fix_id=fix_id, window=window
    )
    return base + "\n" + ave_energy


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLammpsInteractiveSession(unittest.TestCase):
    def _get_potential(self):
        return get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )

    def test_get_energy_pot_matches_direct_getter(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = self._get_potential()
        lmp = lammps_run(
            structure=structure,
            potential_dataframe=df_pot_selected,
            input_template=_nvt_template(),
        )
        energy = get_energy_pot_with_lammpslib(lmp=lmp, run=10)
        self.assertIsInstance(energy, float)
        self.assertAlmostEqual(energy, lmp.interactive_energy_pot_getter(), places=8)
        lammps_shutdown(lmp_instance=lmp)

    def test_get_energy_pot_average_is_close_to_instantaneous(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = self._get_potential()
        window = 10
        lmp = lammps_run(
            structure=structure,
            potential_dataframe=df_pot_selected,
            input_template=_nvt_with_average_template(window=window),
        )
        average_energy = get_energy_pot_average_with_lammpslib(lmp=lmp, run=window)
        instantaneous_energy = lmp.interactive_energy_pot_getter()
        self.assertIsInstance(average_energy, float)
        self.assertTrue(np.isfinite(average_energy))
        self.assertNotEqual(average_energy, 0.0)
        self.assertAlmostEqual(average_energy, instantaneous_energy, delta=5.0)
        lammps_shutdown(lmp_instance=lmp)

    def test_snapshot_round_trip(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = self._get_potential()
        lmp = lammps_run(
            structure=structure,
            potential_dataframe=df_pot_selected,
            input_template=_nvt_template(),
        )
        get_energy_pot_with_lammpslib(lmp=lmp, run=10)
        snapshot = get_structure_snapshot_with_lammpslib(structure=structure, lmp=lmp)

        self.assertEqual(
            list(snapshot.get_chemical_symbols()), list(structure.get_chemical_symbols())
        )
        np.testing.assert_allclose(snapshot.cell.array, lmp.interactive_cells_getter())
        np.testing.assert_allclose(snapshot.positions, lmp.interactive_positions_getter())
        np.testing.assert_allclose(
            snapshot.get_velocities(), lmp.interactive_velocities_getter()
        )
        energy_before_restore = lmp.interactive_energy_pot_getter()
        lammps_shutdown(lmp_instance=lmp)

        # Resume from the snapshot without re-randomising velocities: a fresh
        # `velocity create` would overwrite the velocities the snapshot carries.
        resume_template = _nvt_template(include_velocity=False)
        lmp_resumed = lammps_run(
            structure=snapshot,
            potential_dataframe=df_pot_selected,
            input_template=resume_template,
        )
        energy_after_restore = get_energy_pot_with_lammpslib(lmp=lmp_resumed, run=0)
        self.assertAlmostEqual(energy_before_restore, energy_after_restore, places=6)
        lammps_shutdown(lmp_instance=lmp_resumed)
