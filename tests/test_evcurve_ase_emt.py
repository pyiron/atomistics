from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import (
    EnergyVolumeCurveWorkflow,
    optimize_volume,
)


class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        structure = bulk("Al", cubic=True)
        task_dict = optimize_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
        )
        workflow = EnergyVolumeCurveWorkflow(
            structure=result_dict["structure_with_optimized_volume"],
            num_points=11,
            fit_type="polynomial",
            fit_order=3,
            vol_range=0.05,
            axes=("x", "y", "z"),
            strains=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(task_dict=task_dict, ase_calculator=EMT())
        fit_dict = workflow.analyse_structures(output_dict=result_dict)
        thermal_properties_dict = workflow.get_thermal_properties(
            temperatures=[100, 1000], output_keys=["temperatures", "volumes"]
        )
        temperatures_ev, volumes_ev = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        self.assertAlmostEqual(fit_dict["volume_eq"], 63.72747170239313)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 39.51954433668759)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 2.148388436768747)
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])
