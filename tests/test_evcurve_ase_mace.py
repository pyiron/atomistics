from ase.build import bulk
from ase.optimize import LBFGS
import numpy as np
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import optimize_positions_and_volume, EnergyVolumeCurveWorkflow


try:
    from mace.calculators import mace_mp

    skip_mace_test = False
except ImportError:
    skip_mace_test = True


@unittest.skipIf(
    skip_mace_test, "mace is not installed, so the mace tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        structure = bulk("Al", cubic=True)
        ase_calculator = mace_mp(
            model="medium",
            dispersion=False,
            default_dtype="float32",
            device='cpu'
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=ase_calculator,
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.001}
        )
        workflow = EnergyVolumeCurveWorkflow(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            num_points=11,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=('x', 'y', 'z'),
            strains=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=ase_calculator,
        )
        fit_dict = workflow.analyse_structures(output_dict=result_dict)
        thermal_properties_dict = workflow.get_thermal_properties(
            temperatures=[100, 1000],
            output_keys=["temperatures", "volumes"]
        )
        temperatures_ev, volumes_ev = thermal_properties_dict["temperatures"], thermal_properties_dict["volumes"]
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 66.94655948308437, atol=1e-04))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 64.40241949760645, atol=1e-02))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 4.460574503792641, atol=1e-02))
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])
