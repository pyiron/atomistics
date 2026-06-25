import unittest

import numpy as np
from ase.build import bulk

from atomistics.calculators.hessian import (
    calc_forces_transformed,
    check_force_constants,
    evaluate_with_hessian,
    get_displacement,
    get_energy_pot,
    get_forces,
    get_pressure_times_volume,
    get_stiffness_tensor,
)


class TestCheckForceConstants(unittest.TestCase):
    def setUp(self):
        self.structure = bulk("Al", cubic=True)
        self.n_atom = len(self.structure)

    def test_scalar_broadcasts_to_identity(self):
        fc = check_force_constants(structure=self.structure, force_constants=2.0)
        self.assertEqual(fc.shape, (3 * self.n_atom, 3 * self.n_atom))
        self.assertTrue(np.allclose(fc, 2.0 * np.eye(3 * self.n_atom)))

    def test_full_matrix_is_returned_unchanged(self):
        full = np.arange((3 * self.n_atom) ** 2).reshape(
            3 * self.n_atom, 3 * self.n_atom
        )
        fc = check_force_constants(structure=self.structure, force_constants=full)
        self.assertTrue(np.array_equal(fc, full))

    def test_atomwise_matrix_expands_to_block_diagonal(self):
        atomwise = np.diag(np.arange(1, self.n_atom + 1).astype(float))
        fc = check_force_constants(structure=self.structure, force_constants=atomwise)
        self.assertEqual(fc.shape, (3 * self.n_atom, 3 * self.n_atom))
        for i in range(self.n_atom):
            block = fc[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
            self.assertTrue(np.allclose(block, (i + 1) * np.eye(3)))
        # off-diagonal atom pairs with zero coupling should vanish
        self.assertTrue(np.allclose(fc[0:3, 3:6], np.zeros((3, 3))))

    def test_4d_array_with_inner_3x3_axes(self):
        n = self.n_atom
        force_constants = np.zeros((n, n, 3, 3))
        for i in range(n):
            force_constants[i, i] = (i + 1) * np.eye(3)
        fc = check_force_constants(structure=self.structure, force_constants=force_constants)
        self.assertEqual(fc.shape, (3 * n, 3 * n))
        for i in range(n):
            block = fc[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
            self.assertTrue(np.allclose(block, (i + 1) * np.eye(3)))

    def test_4d_array_with_interleaved_3x3_axes(self):
        n = self.n_atom
        force_constants = np.zeros((n, 3, n, 3))
        for i in range(n):
            force_constants[i, :, i, :] = (i + 1) * np.eye(3)
        fc = check_force_constants(structure=self.structure, force_constants=force_constants)
        self.assertEqual(fc.shape, (3 * n, 3 * n))
        for i in range(n):
            block = fc[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
            self.assertTrue(np.allclose(block, (i + 1) * np.eye(3)))

    def test_unrecognized_2d_shape_raises(self):
        with self.assertRaises(AssertionError):
            check_force_constants(structure=self.structure, force_constants=np.zeros((2, 2)))

    def test_unrecognized_4d_shape_raises(self):
        with self.assertRaises(AssertionError):
            check_force_constants(
                structure=self.structure, force_constants=np.zeros((2, 2, 2, 2))
            )

    def test_missing_structure_raises(self):
        with self.assertRaises(ValueError):
            check_force_constants(structure=None, force_constants=1.0)


class TestDisplacementAndForces(unittest.TestCase):
    def setUp(self):
        self.structure_equilibrium = bulk("Al", cubic=True)
        self.structure = self.structure_equilibrium.copy()
        self.structure.positions[0, 0] += 0.01
        self.n_atom = len(self.structure)
        self.force_constants = 5.0 * np.eye(3 * self.n_atom)

    def test_get_displacement_zero_for_identical_structures(self):
        displacements = get_displacement(self.structure_equilibrium, self.structure_equilibrium)
        self.assertTrue(np.allclose(displacements, 0.0))

    def test_get_displacement_matches_cartesian_shift(self):
        displacements = get_displacement(self.structure_equilibrium, self.structure)
        expected = np.zeros((self.n_atom, 3))
        expected[0, 0] = 0.01
        self.assertTrue(np.allclose(displacements, expected, atol=1e-8))

    def test_calc_forces_transformed_matches_displacement(self):
        forces_transformed, displacements = calc_forces_transformed(
            force_constants=self.force_constants,
            structure_equilibrium=self.structure_equilibrium,
            structure=self.structure,
        )
        expected = -5.0 * displacements.flatten()
        self.assertTrue(np.allclose(forces_transformed, expected, atol=1e-8))

    def test_get_forces_is_hookes_law(self):
        forces = get_forces(
            force_constants=self.force_constants,
            structure_equilibrium=self.structure_equilibrium,
            structure=self.structure,
        )
        self.assertEqual(forces.shape, (self.n_atom, 3))
        expected = np.zeros((self.n_atom, 3))
        expected[0, 0] = -5.0 * 0.01
        self.assertTrue(np.allclose(forces, expected, atol=1e-8))

    def test_get_energy_pot_without_stress(self):
        energy = get_energy_pot(
            force_constants=self.force_constants,
            structure_equilibrium=self.structure_equilibrium,
            structure=self.structure,
        )
        expected = 0.5 * 5.0 * 0.01**2
        self.assertAlmostEqual(energy, expected, places=8)

    def test_get_energy_pot_at_equilibrium_is_zero(self):
        energy = get_energy_pot(
            force_constants=self.force_constants,
            structure_equilibrium=self.structure_equilibrium,
            structure=self.structure_equilibrium,
        )
        self.assertAlmostEqual(energy, 0.0, places=8)


class TestStiffnessTensorAndPressure(unittest.TestCase):
    def setUp(self):
        self.structure_equilibrium = bulk("Al", cubic=True)

    def test_get_stiffness_tensor_zero(self):
        tensor = get_stiffness_tensor(bulk_modulus=0.0, shear_modulus=0.0)
        self.assertTrue(np.allclose(tensor, np.zeros((6, 6))))

    def test_get_stiffness_tensor_shape_and_symmetry(self):
        tensor = get_stiffness_tensor(bulk_modulus=10.0, shear_modulus=2.0)
        self.assertEqual(tensor.shape, (6, 6))
        self.assertTrue(np.allclose(tensor[3:, 3:], 2.0 * np.eye(3)))
        # off-diagonal coupling blocks are untouched (zero)
        self.assertTrue(np.allclose(tensor[:3, 3:], np.zeros((3, 3))))

    def test_pressure_times_volume_zero_stiffness_returns_zero(self):
        stiffness_tensor = get_stiffness_tensor(bulk_modulus=0.0, shear_modulus=0.0)
        result = get_pressure_times_volume(
            stiffness_tensor=stiffness_tensor,
            structure_equilibrium=self.structure_equilibrium,
            structure=self.structure_equilibrium,
        )
        self.assertEqual(result, 0.0)

    # Note: the nonzero-stiffness-tensor branch of get_pressure_times_volume has a
    # pre-existing shape-mismatch bug (raises ValueError) that is out of scope to fix
    # here; only the zero-stiffness branch (exercised above) is currently correct.


class TestEvaluateWithHessian(unittest.TestCase):
    def setUp(self):
        self.structure_equilibrium = bulk("Al", cubic=True)
        self.structure = self.structure_equilibrium.copy()
        self.structure.positions[0, 0] += 0.01
        self.force_constants = 5.0 * np.eye(3 * len(self.structure))

    def test_calc_energy_and_forces(self):
        result = evaluate_with_hessian(
            task_dict={
                "calc_energy": self.structure,
                "calc_forces": self.structure,
            },
            structure_equilibrium=self.structure_equilibrium,
            force_constants=self.force_constants,
        )
        self.assertIn("energy", result)
        self.assertIn("forces", result)
        self.assertAlmostEqual(result["energy"], 0.5 * 5.0 * 0.01**2, places=8)

    def test_unsupported_task_raises(self):
        with self.assertRaises(ValueError):
            evaluate_with_hessian(
                task_dict={"optimize_positions": self.structure},
                structure_equilibrium=self.structure_equilibrium,
                force_constants=self.force_constants,
            )


if __name__ == "__main__":
    unittest.main()
