from ase.build import bulk
from ase.optimize import LBFGS
import numpy as np
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import (
    optimize_positions_and_volume,
    get_tasks_for_energy_volume_curve,
    analyse_results_for_energy_volume_curve,
)


try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.atomic_system import SystemConfig
    from orb_models.forcefield.calculator import ORBCalculator

    skip_orb_test = False
except ImportError:
    skip_orb_test = True


@unittest.skipIf(
    skip_orb_test, "orb-models is not installed, so the orb-model tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        structure = bulk("Al", cubic=True)
        orb_model = pretrained.ORB_PRETRAINED_MODELS['orb-v2']  # Get the model
        model = orb_model(device="cpu")
        system_config = SystemConfig(radius=10.0, max_num_neighbors=20)
        ase_calculator = ORBCalculator(
            model=model,
            device="cpu",
            system_config=system_config
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=ase_calculator,
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.001},
        )
        task_dict = get_tasks_for_energy_volume_curve(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            num_points=11,
            vol_range=0.05,
            axes=("x", "y", "z"),
        )
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=ase_calculator,
        )
        fit_dict = analyse_results_for_energy_volume_curve(
            output_dict=result_dict,
            task_dict=task_dict,
            fit_type="polynomial",
            fit_order=3,
        )
        self.assertTrue(
            np.isclose(fit_dict["volume_eq"], 66.6180623643703, atol=1e-04)
        )
        self.assertTrue(
            np.isclose(fit_dict["bulkmodul_eq"], 125.44056924108797, atol=1e-02)
        )
        self.assertTrue(
            np.isclose(fit_dict["b_prime_eq"], -16.372295442280702, atol=1e-02)
        )
