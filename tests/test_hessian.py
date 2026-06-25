import unittest

import numpy as np
from ase.atoms import Atoms

from atomistics.calculators.hessian import (
    get_energy_pot,
    get_pressure_times_volume,
    get_stiffness_tensor,
)


def _cubic_atoms(cell: np.ndarray) -> Atoms:
    return Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)


class TestGetPressureTimesVolume(unittest.TestCase):
    def setUp(self):
        self.a = 4.0
        self.cell_equilibrium = np.eye(3) * self.a
        self.structure_equilibrium = _cubic_atoms(self.cell_equilibrium)

    def test_zero_stiffness_tensor_returns_zero(self):
        strained_cell = self.cell_equilibrium * 1.05
        structure = _cubic_atoms(strained_cell)
        result = get_pressure_times_volume(
            stiffness_tensor=np.zeros((6, 6)),
            structure_equilibrium=self.structure_equilibrium,
            structure=structure,
        )
        self.assertEqual(result, 0.0)

    def test_hydrostatic_strain_matches_closed_form(self):
        bulk_modulus = 5.0
        delta = 0.01
        structure = _cubic_atoms(self.cell_equilibrium * (1 + delta))
        stiffness_tensor = get_stiffness_tensor(
            bulk_modulus=bulk_modulus, shear_modulus=0.0
        )
        result = get_pressure_times_volume(
            stiffness_tensor=stiffness_tensor,
            structure_equilibrium=self.structure_equilibrium,
            structure=structure,
        )
        # For an isotropic hydrostatic strain eps = delta * I, the stiffness
        # tensor's shear contribution cancels and pressure_voigt[:3] reduces
        # to -3 * bulk_modulus * delta, giving -Tr(pressure @ eps) = 9 *
        # bulk_modulus * delta**2.
        expected = 9 * bulk_modulus * delta**2 * structure.get_volume()
        self.assertAlmostEqual(result, expected)

    def test_shear_cross_term_matches_closed_form(self):
        # Isolate the off-diagonal Voigt coupling between the xy and yz
        # shear components to confirm the Voigt ordering convention
        # (xx, yy, zz, yz, xz, xy) used elsewhere in the codebase.
        g_xy = 0.02
        g_yz = -0.015
        coupling = 3.0
        epsilon_target = np.array(
            [
                [0.0, g_xy, 0.0],
                [g_xy, 0.0, g_yz],
                [0.0, g_yz, 0.0],
            ]
        )
        structure = _cubic_atoms(self.cell_equilibrium @ (np.eye(3) + epsilon_target))

        stiffness_tensor = np.zeros((6, 6))
        stiffness_tensor[3, 5] = coupling
        stiffness_tensor[5, 3] = coupling

        result = get_pressure_times_volume(
            stiffness_tensor=stiffness_tensor,
            structure_equilibrium=self.structure_equilibrium,
            structure=structure,
        )
        expected = 4 * coupling * g_xy * g_yz * structure.get_volume()
        self.assertAlmostEqual(result, expected)

    def test_get_energy_pot_matches_pressure_times_volume_without_displacement(self):
        bulk_modulus = 7.0
        shear_modulus = 2.0
        delta = 0.02
        structure = _cubic_atoms(self.cell_equilibrium * (1 + delta))
        force_constants = np.zeros((3, 3))

        energy = get_energy_pot(
            force_constants=force_constants,
            structure_equilibrium=self.structure_equilibrium,
            structure=structure,
            bulk_modulus=bulk_modulus,
            shear_modulus=shear_modulus,
        )
        expected = get_pressure_times_volume(
            stiffness_tensor=get_stiffness_tensor(
                bulk_modulus=bulk_modulus, shear_modulus=shear_modulus
            ),
            structure_equilibrium=self.structure_equilibrium,
            structure=structure,
        )
        self.assertAlmostEqual(energy, expected)


if __name__ == "__main__":
    unittest.main()
