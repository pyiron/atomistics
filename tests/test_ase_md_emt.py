from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from atomistics.calculators import (
    calc_molecular_dynamics_langevin_with_ase,
    calc_molecular_dynamics_npt_with_ase,
)
import numpy as np
import unittest


def get_volume(cell):
    return np.abs(np.linalg.det(cell))


class TestASEMD(unittest.TestCase):
    def test_ase_langevin(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        result_dict = calc_molecular_dynamics_langevin_with_ase(
            structure=structure,
            ase_calculator=EMT(),
            run=100,
            thermo=10,
            timestep=1 * units.fs,
            temperature=100,
            friction=0.002,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertEqual(result_dict["energy_pot"].shape, (10,))
        self.assertEqual(result_dict["energy_tot"].shape, (10,))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 25)
        self.assertTrue(result_dict["temperature"][-1] < 75)

    def test_ase_npt(self):
        structure = bulk("Al", a=3.5, cubic=True).repeat([2, 2, 2])
        result_dict = calc_molecular_dynamics_npt_with_ase(
            structure=structure,
            ase_calculator=EMT(),
            run=100,
            thermo=10,
            timestep=1 * units.fs,
            ttime=100 * units.fs,
            pfactor=2e6 * units.GPa * (units.fs**2),
            temperature=100,
            externalstress=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10,))
        self.assertEqual(result_dict["energy_pot"].shape, (10,))
        self.assertEqual(result_dict["energy_tot"].shape, (10,))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 50)
        self.assertTrue(result_dict["temperature"][-1] < 100)
        self.assertTrue(
            get_volume(result_dict["cell"][0]) < get_volume(result_dict["cell"][-1])
        )
