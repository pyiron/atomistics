from ase.build import bulk
import numpy as np
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import (
    get_thermal_properties_for_energy_volume_curve,
    get_tasks_for_energy_volume_curve,
    analyse_results_for_energy_volume_curve,
)

try:
    from gpaw import GPAW, PW

    skip_gpaw_test = False
except ImportError:
    skip_gpaw_test = True


@unittest.skipIf(
    skip_gpaw_test, "gpaw is not installed, so the gpaw tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        structure = bulk("Al", a=4.05, cubic=True)
        task_dict = get_tasks_for_energy_volume_curve(
            structure=structure,
            num_points=11,
            vol_range=0.05,
            axes=("x", "y", "z"),
        )
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=GPAW(xc="PBE", mode=PW(300), kpts=(3, 3, 3)),
        )
        fit_dict = analyse_results_for_energy_volume_curve(
            output_dict=result_dict,
            task_dict=task_dict,
            fit_type="polynomial",
            fit_order=3,
        )
        thermal_properties_dict = get_thermal_properties_for_energy_volume_curve(
            fit_dict=fit_dict,
            masses=structure.get_masses(),
            temperatures=[100, 1000],
            output_keys=["temperatures", "volumes"],
        )
        temperatures_ev, volumes_ev = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        self.assertTrue(
            np.isclose(fit_dict["volume_eq"], 66.44252286131331, atol=1e-04)
        )
        self.assertTrue(
            np.isclose(fit_dict["bulkmodul_eq"], 72.38919826652857, atol=1e-04)
        )
        self.assertTrue(
            np.isclose(fit_dict["b_prime_eq"], 4.453836551712821, atol=1e-04)
        )
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])
