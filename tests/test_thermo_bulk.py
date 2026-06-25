import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk

from atomistics.workflows.evcurve.debye import DebyeModel
from atomistics.workflows.evcurve.thermo import ThermoBulk, get_thermo_bulk_model


def _get_thermo_bulk_model():
    volumes = [
        63.10883669478296,
        63.77314023893856,
        64.43744378309412,
        65.10174732724975,
        65.7660508714054,
        66.43035441556098,
        67.09465795971657,
        67.7589615038722,
        68.42326504802779,
        69.08756859218344,
        69.75187213633905,
    ]
    return get_thermo_bulk_model(
        temperatures=np.arange(1.0, 1550.0, 50.0),
        debye_model=DebyeModel(
            fit_dict={
                "energy_eq": -13.439961346483864,
                "volume_eq": 66.43032532814925,
                "bulkmodul_eq": 77.61265363975706,
                "b_prime_eq": 1.2734991618131122,
                "volume": volumes,
                "fit_dict": {
                    "fit_type": "vinet",
                },
            },
            masses=bulk("Al", cubic=True).get_masses(),
            num_steps=50,
        ),
    )


class TestThermoBulkValidation(unittest.TestCase):
    def setUp(self):
        self.thermo = ThermoBulk()

    def test_temperatures_unset_raises(self):
        with self.assertRaises(ValueError):
            _ = self.thermo.temperatures

    def test_temperatures_setter_requires_list_like(self):
        with self.assertRaises(ValueError):
            self.thermo.temperatures = 300.0

    def test_volumes_unset_raises(self):
        with self.assertRaises(ValueError):
            _ = self.thermo.volumes

    def test_volumes_setter_requires_list_like(self):
        with self.assertRaises(ValueError):
            self.thermo.volumes = 10.0

    def test_energies_unset_raises(self):
        with self.assertRaises(ValueError):
            _ = self.thermo.energies


class TestThermoBulkEnergiesSetter(unittest.TestCase):
    def setUp(self):
        self.thermo = ThermoBulk()
        self.thermo.temperatures = np.linspace(0, 100, 5)
        self.thermo.volumes = np.linspace(10, 20, 3)

    def test_energies_2d_passthrough(self):
        energies = np.arange(15).reshape(5, 3).astype(float)
        self.thermo.energies = energies
        self.assertTrue(np.array_equal(self.thermo.energies, energies))

    def test_energies_1d_matching_volumes_is_tiled_per_temperature(self):
        per_volume = np.array([1.0, 2.0, 3.0])
        self.thermo.energies = per_volume
        self.assertEqual(self.thermo.energies.shape, (5, 3))
        for row in self.thermo.energies:
            self.assertTrue(np.array_equal(row, per_volume))

    def test_energies_1d_mismatched_length_raises(self):
        with self.assertRaises(ValueError):
            self.thermo.energies = np.array([1.0, 2.0])

    def test_energies_scalar_broadcasts_with_temperature_volume_convention(self):
        self.thermo.energies = -2.0
        self.assertEqual(self.thermo.energies.shape, (5, 3))
        self.assertTrue(np.allclose(self.thermo.energies, -2.0))


class TestThermoBulkSetters(unittest.TestCase):
    def setUp(self):
        self.thermo = ThermoBulk()

    def test_set_temperatures_defaults(self):
        self.thermo.set_temperatures()
        self.assertEqual(len(self.thermo.temperatures), 50)
        self.assertEqual(self.thermo.temperatures[0], 0.0)
        self.assertEqual(self.thermo.temperatures[-1], 1500.0)

    def test_set_volumes_default_max_is_1_1_times_min(self):
        self.thermo.set_volumes(volume_min=10.0, volume_steps=5)
        self.assertEqual(len(self.thermo.volumes), 5)
        self.assertAlmostEqual(self.thermo.volumes[0], 10.0)
        self.assertAlmostEqual(self.thermo.volumes[-1], 11.0)

    def test_set_volumes_explicit_max(self):
        self.thermo.set_volumes(volume_min=10.0, volume_max=20.0, volume_steps=3)
        self.assertTrue(np.allclose(self.thermo.volumes, [10.0, 15.0, 20.0]))

    def test_meshgrid_shapes(self):
        self.thermo.set_temperatures(temperature_steps=4)
        self.thermo.set_volumes(volume_min=10.0, volume_max=20.0, volume_steps=3)
        x_grid, y_grid = self.thermo.meshgrid()
        self.assertEqual(x_grid.shape, (4, 3))
        self.assertEqual(y_grid.shape, (4, 3))


class TestThermoBulkMinimumEnergyPath(unittest.TestCase):
    def test_linear_energies_have_no_minimum(self):
        thermo = ThermoBulk()
        thermo._fit_order = 1
        thermo.temperatures = np.linspace(0, 100, 5)
        thermo.volumes = np.linspace(10, 20, 3)
        thermo.energies = np.array([1.0, 0.5, 0.0])
        path = thermo.get_minimum_energy_path()
        self.assertTrue(np.all(np.isnan(path)))

    def test_pressure_not_implemented(self):
        thermo = ThermoBulk()
        thermo.temperatures = np.linspace(0, 100, 5)
        thermo.volumes = np.linspace(10, 20, 3)
        thermo.energies = np.array([1.0, 0.5, 0.0])
        with self.assertRaises(NotImplementedError):
            thermo.get_minimum_energy_path(pressure=1.0)


class TestThermoBulkFreeEnergyAndInterpolation(unittest.TestCase):
    def setUp(self):
        self.thermo = _get_thermo_bulk_model()

    def test_get_free_energy_matches_polyval(self):
        vol = self.thermo.volumes[0]
        free_energy = self.thermo.get_free_energy(vol)
        expected = np.polyval(self.thermo._coeff, vol)
        self.assertTrue(np.allclose(free_energy, expected))

    def test_get_free_energy_with_pressure_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.thermo.get_free_energy(self.thermo.volumes[0], pressure=1.0)

    def test_interpolate_volume_returns_new_object_with_consistent_shape(self):
        new_volumes = self.thermo.volumes[:5]
        interpolated = self.thermo.interpolate_volume(new_volumes)
        self.assertIsNot(interpolated, self.thermo)
        self.assertEqual(len(interpolated.volumes), 5)
        self.assertEqual(
            interpolated.energies.shape, (len(self.thermo.temperatures), 5)
        )

    def test_interpolate_volume_with_explicit_fit_order(self):
        interpolated = self.thermo.interpolate_volume(
            self.thermo.volumes[:3], fit_order=2
        )
        self.assertEqual(self.thermo._fit_order, 2)
        self.assertEqual(len(interpolated.volumes), 3)

    def test_get_free_energy_p_shape(self):
        free_energy_p = self.thermo.get_free_energy_p()
        self.assertEqual(free_energy_p.shape, self.thermo.temperatures.shape)

    def test_get_entropy_p_shape(self):
        entropy_p = self.thermo.get_entropy_p()
        self.assertEqual(entropy_p.shape, self.thermo.temperatures.shape)

    def test_get_entropy_v_shape(self):
        entropy_v = self.thermo.get_entropy_v()
        self.assertEqual(entropy_v.shape, self.thermo.temperatures.shape)


class TestThermoBulkPlotting(unittest.TestCase):
    def setUp(self):
        self.thermo = _get_thermo_bulk_model()

    def tearDown(self):
        plt.close("all")

    def test_plot_free_energy(self):
        self.thermo.plot_free_energy()

    def test_plot_entropy(self):
        self.thermo.plot_entropy()

    def test_plot_heat_capacity_in_kB(self):
        self.thermo.plot_heat_capacity(to_kB=True)

    def test_plot_heat_capacity_in_joules(self):
        self.thermo.plot_heat_capacity(to_kB=False)

    def test_contour_pressure(self):
        self.thermo.contour_pressure()

    def test_contour_entropy(self):
        self.thermo.contour_entropy()

    def test_plot_contourf_creates_axes_when_none_given(self):
        ax = self.thermo.plot_contourf()
        self.assertIsNotNone(ax)

    def test_plot_contourf_with_existing_axes_and_min_erg_path(self):
        fig, ax = plt.subplots(1, 1)
        result = self.thermo.plot_contourf(ax=ax, show_min_erg_path=True)
        self.assertIs(result, ax)

    def test_plot_min_energy_path_creates_axes_when_none_given(self):
        ax = self.thermo.plot_min_energy_path()
        self.assertIsNotNone(ax)

    def test_plot_min_energy_path_with_existing_axes(self):
        fig, ax = plt.subplots(1, 1)
        result = self.thermo.plot_min_energy_path(ax=ax)
        self.assertIs(result, ax)


if __name__ == "__main__":
    unittest.main()
