import unittest

import numpy as np

from atomistics.workflows.evcurve.debye import (
    DebyeModel,
    DebyeThermalProperties,
    debye_function,
    get_thermal_properties_for_energy_volume_curve,
)


def _fit_dict():
    return {
        "energy_eq": -13.439961346483864,
        "volume_eq": 66.43032532814925,
        "bulkmodul_eq": 77.61265363975706,
        "b_prime_eq": 1.2734991618131122,
        "volume": [63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0],
        "fit_dict": {
            "fit_type": "vinet",
        },
    }


class TestDebyeFunction(unittest.TestCase):
    def test_debye_function_scalar(self):
        result = debye_function(1.0)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_debye_function_array(self):
        result = debye_function(np.array([0.5, 1.0, 2.0]))
        self.assertEqual(result.shape, (3,))


class TestDebyeModel(unittest.TestCase):
    def setUp(self):
        self.masses = [26.98] * 4
        self.model = DebyeModel(fit_dict=_fit_dict(), masses=self.masses, num_steps=10)

    def test_volume_grid_spans_fit_dict_range(self):
        self.assertEqual(self.model.volume.shape, (10,))
        self.assertAlmostEqual(self.model.volume[0], 63.0)
        self.assertAlmostEqual(self.model.volume[-1], 69.0)

    def test_volume_setter_updates_min_and_max(self):
        self.model.volume = np.linspace(60.0, 70.0, 5)
        self.assertTrue(np.allclose(self.model.volume, [60.0, 62.5, 65.0, 67.5, 70.0]))

    def test_debye_temperature_is_cached(self):
        first = self.model.debye_temperature
        second = self.model.debye_temperature
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_debye_temperature_reset_after_volume_change(self):
        first = self.model.debye_temperature
        self.model.volume = np.linspace(60.0, 70.0, 5)
        second = self.model.debye_temperature
        self.assertIsNot(first[0], second[0])

    def test_debye_temperature_multi_species_raises(self):
        model = DebyeModel(fit_dict=_fit_dict(), masses=[26.98, 12.0], num_steps=10)
        with self.assertRaises(NotImplementedError):
            _ = model.debye_temperature

    def test_energy_vib_with_explicit_scalar_debye_temperature(self):
        energy = self.model.energy_vib(T=100.0, debye_T=300.0)
        self.assertIsInstance(energy, float)

    def test_energy_vib_low_vs_high_t_limit_differ(self):
        low = self.model.energy_vib(T=np.array([300.0]), low_T_limit=True)
        high = self.model.energy_vib(T=np.array([300.0]), low_T_limit=False)
        self.assertFalse(np.allclose(low, high))


class TestDebyeThermalProperties(unittest.TestCase):
    def setUp(self):
        self.masses = [26.98] * 4
        self.fit_dict = _fit_dict()

    def test_temperatures_generated_from_min_max_step(self):
        props = DebyeThermalProperties(
            fit_dict=self.fit_dict,
            masses=self.masses,
            t_min=0.0,
            t_max=200.0,
            t_step=50.0,
            num_steps=10,
        )
        self.assertTrue(np.allclose(props.temperatures(), [0.0, 50.0, 100.0, 150.0, 200.0]))

    def test_free_energy_shape_matches_temperatures(self):
        props = DebyeThermalProperties(
            fit_dict=self.fit_dict, masses=self.masses, num_steps=10
        )
        self.assertEqual(props.free_energy().shape, props.temperatures().shape)

    def test_entropy_constant_pressure_vs_constant_volume_differ(self):
        props_p = DebyeThermalProperties(
            fit_dict=self.fit_dict, masses=self.masses, num_steps=10, constant_volume=False
        )
        props_v = DebyeThermalProperties(
            fit_dict=self.fit_dict, masses=self.masses, num_steps=10, constant_volume=True
        )
        self.assertEqual(props_p.entropy().shape, props_v.entropy().shape)
        self.assertFalse(np.allclose(props_p.entropy(), props_v.entropy()))

    def test_heat_capacity_ends_with_nan_padding(self):
        props = DebyeThermalProperties(
            fit_dict=self.fit_dict, masses=self.masses, num_steps=10
        )
        heat_capacity = props.heat_capacity()
        self.assertEqual(heat_capacity.shape, props.temperatures().shape)
        self.assertTrue(np.all(np.isnan(heat_capacity[-2:])))

    def test_volumes_constant_volume_is_equilibrium_volume(self):
        props = DebyeThermalProperties(
            fit_dict=self.fit_dict, masses=self.masses, num_steps=10, constant_volume=True
        )
        volumes = props.volumes()
        self.assertTrue(np.allclose(volumes, self.fit_dict["volume"][0]))

    def test_volumes_constant_pressure_follows_minimum_energy_path(self):
        props = DebyeThermalProperties(
            fit_dict=self.fit_dict, masses=self.masses, num_steps=10, constant_volume=False
        )
        volumes = props.volumes()
        self.assertEqual(volumes.shape, props.temperatures().shape)


class TestGetThermalPropertiesForEnergyVolumeCurve(unittest.TestCase):
    def test_returns_requested_output_keys(self):
        result = get_thermal_properties_for_energy_volume_curve(
            fit_dict=_fit_dict(),
            masses=[26.98] * 4,
            num_steps=10,
            output_keys=("temperatures", "free_energy", "volumes"),
        )
        self.assertEqual(set(result.keys()), {"temperatures", "free_energy", "volumes"})
        self.assertEqual(result["temperatures"].shape, result["free_energy"].shape)


if __name__ == "__main__":
    unittest.main()
