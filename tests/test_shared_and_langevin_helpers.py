from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from ase.build import bulk
import numpy as np
import unittest

from atomistics.shared.parallel import (
    _convert_task_dict_to_task_lst,
    _convert_task_lst_to_task_dict,
    evaluate_with_parallel_executor,
)
from atomistics.shared.thermal_expansion import (
    ThermalExpansionProperties,
    get_thermal_expansion_output,
)
from atomistics.shared.tqdm_iterator import get_tqdm_iterator
import atomistics.shared.tqdm_iterator as tqdm_iterator_module
from atomistics.workflows.langevin import (
    EV_TO_U_ANGSQ_PER_FSSQ,
    LangevinWorkflow,
    convert_to_acceleration,
    get_first_half_step,
    get_initial_velocities,
    langevin_delta_v,
)
from atomistics.workflows.molecular_dynamics import (
    calc_molecular_dynamics_thermal_expansion,
)


class TestSharedHelpers(unittest.TestCase):
    def test_convert_task_dict_to_task_list(self):
        self.assertEqual(
            _convert_task_dict_to_task_lst({"calc": {1: "a", 2: "b"}, "other": "c"}),
            [{"calc": {1: "a"}}, {"calc": {2: "b"}}, {"other": "c"}],
        )

    def test_convert_task_list_to_task_dict(self):
        self.assertEqual(
            _convert_task_lst_to_task_dict(
                [{"calc": {1: "a"}}, {"calc": {2: "b"}}, {"other": "c"}]
            ),
            {"calc": {1: "a", 2: "b"}, "other": "c"},
        )

    def test_evaluate_with_parallel_executor(self):
        def evaluate_function(task_dict: dict, shift: int) -> dict:
            result = {}
            for key, value in task_dict.items():
                if isinstance(value, dict):
                    result[key] = {k: v + shift for k, v in value.items()}
                else:
                    result[key] = value + shift
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = evaluate_with_parallel_executor(
                evaluate_function=evaluate_function,
                task_dict={"calc": {1: 2, 3: 4}, "other": 5},
                executor=executor,
                shift=10,
            )
        self.assertEqual(result, {"calc": {1: 12, 3: 14}, "other": 15})

    def test_thermal_expansion_properties_and_output(self):
        temperatures = np.array([100.0, 200.0])
        volumes = np.array([10.0, 11.0])
        thermal = ThermalExpansionProperties(
            temperatures_lst=temperatures, volumes_lst=volumes
        )
        self.assertTrue(np.array_equal(thermal.temperatures(), temperatures))
        self.assertTrue(np.array_equal(thermal.volumes(), volumes))
        output = get_thermal_expansion_output(
            temperatures_lst=temperatures,
            volumes_lst=volumes,
            output_keys=("temperatures", "volumes"),
        )
        self.assertTrue(np.array_equal(output["temperatures"], temperatures))
        self.assertTrue(np.array_equal(output["volumes"], volumes))

    def test_get_tqdm_iterator_without_tqdm(self):
        lst = [1, 2, 3]
        with patch.object(tqdm_iterator_module, "tqdm_available", False):
            self.assertIs(get_tqdm_iterator(lst), lst)

    def test_get_tqdm_iterator_with_tqdm(self):
        lst = [1, 2, 3]
        with (
            patch.object(tqdm_iterator_module, "tqdm_available", True),
            patch.object(
                tqdm_iterator_module, "tqdm", side_effect=lambda x: iter(x), create=True
            ),
        ):
            self.assertEqual(list(get_tqdm_iterator(lst)), lst)

    def test_calc_molecular_dynamics_thermal_expansion(self):
        structure = bulk("Al", cubic=True)
        task_dict = calc_molecular_dynamics_thermal_expansion(structure=structure)
        self.assertEqual(list(task_dict.keys()), ["calc_molecular_dynamics_thermal_expansion"])
        self.assertIs(task_dict["calc_molecular_dynamics_thermal_expansion"], structure)


class TestLangevinHelpers(unittest.TestCase):
    def test_convert_to_acceleration(self):
        forces = np.array([[1.0, 2.0, 3.0]])
        masses = np.array([[2.0]])
        acceleration = convert_to_acceleration(forces=forces, masses=masses)
        expected = forces * EV_TO_U_ANGSQ_PER_FSSQ / masses
        self.assertTrue(np.allclose(acceleration, expected))

    def test_langevin_delta_v_without_damping(self):
        self.assertEqual(
            langevin_delta_v(
                temperature=300.0,
                time_step=1.0,
                masses=np.array([[27.0]]),
                velocities=np.array([[0.0, 0.0, 0.0]]),
                damping_timescale=None,
            ),
            0.0,
        )

    def test_langevin_delta_v_with_damping_has_zero_mean_noise(self):
        velocities = np.array([[0.1, 0.2, 0.3], [0.0, -0.1, 0.2]])
        with patch("atomistics.workflows.langevin.np.random.randn", return_value=np.ones((2, 3))):
            delta_v = langevin_delta_v(
                temperature=300.0,
                time_step=1.0,
                masses=np.array([[27.0], [27.0]]),
                velocities=velocities,
                damping_timescale=100.0,
            )
        drag = -0.5 * velocities / 100.0
        self.assertTrue(np.allclose(delta_v, drag))

    def test_get_initial_velocities_zero_centered(self):
        with patch(
            "atomistics.workflows.langevin.np.random.randn",
            return_value=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        ):
            velocities = get_initial_velocities(
                temperature=500.0, masses=np.array([[27.0], [27.0]])
            )
        self.assertTrue(np.allclose(np.mean(velocities, axis=0), 0.0))

    def test_get_first_half_step(self):
        forces = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([[2.0]])
        velocities = np.array([[0.5, 0.0, 0.0]])
        first_half = get_first_half_step(
            forces=forces, masses=masses, time_step=2.0, velocities=velocities
        )
        self.assertTrue(first_half[0, 0] > velocities[0, 0])

    def test_langevin_workflow_generate_and_analyse(self):
        structure = bulk("Al", cubic=True)
        nat = len(structure)
        with patch(
            "atomistics.workflows.langevin.get_initial_velocities",
            return_value=np.zeros((nat, 3)),
        ):
            workflow = LangevinWorkflow(
                structure=structure,
                temperature=300.0,
                damping_timescale=100.0,
                time_step=1.0,
            )

        initial_tasks = workflow.generate_structures()
        self.assertIs(initial_tasks["calc_forces"][0], structure)
        self.assertIs(initial_tasks["calc_energy"][0], structure)

        workflow.forces = np.ones((nat, 3))
        with patch("atomistics.workflows.langevin.langevin_delta_v", return_value=np.zeros((nat, 3))):
            stepped_tasks = workflow.generate_structures()
            eng_pot, eng_kin = workflow.analyse_structures(
                output_dict={"forces": {0: np.ones((nat, 3)) * 2.0}, "energy": {0: -1.5}}
            )

        self.assertEqual(eng_pot, -1.5)
        self.assertGreater(eng_kin, 0.0)
        self.assertFalse(np.allclose(stepped_tasks["calc_forces"][0].positions, structure.positions))


if __name__ == "__main__":
    unittest.main()
