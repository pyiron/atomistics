import os
import unittest

import numpy as np
from ase.build import bulk
from jinja2 import Template

try:
    from atomistics.calculators.lammps import get_potential_by_name, lammps_run, lammps_shutdown
    from atomistics.calculators.lammps.libcalculator import get_energy_pot_with_lammpslib
    from atomistics.calculators.lammps.commands import (
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
