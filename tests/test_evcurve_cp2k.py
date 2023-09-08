import os
import shutil

from ase.build import bulk
import numpy as np
import unittest

from atomistics.calculators.cp2k_ase.calculator import evaluate_with_cp2k
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow


cp2k_command = "cp2k_shell.sopt"
if shutil.which(cp2k_command) is not None:
    skip_cp2k_test = False
else:
    skip_cp2k_test = True


@unittest.skipIf(
    skip_cp2k_test, "cp2k is not installed, so the cp2k tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        resource_path = os.path.join(os.path.dirname(__file__), "static", "cp2k")
        calculator = EnergyVolumeCurveWorkflow(
            structure=bulk("Al", a=4.05, cubic=True),
            num_points=11,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=['x', 'y', 'z'],
            strains=None,
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_cp2k(
            task_dict=structure_dict,
            command=cp2k_command,
            basis_set_file=os.path.join(resource_path, 'BASIS_SET'),
            basis_set="DZVP-GTH-PADE",
            potential_file=os.path.join(resource_path, 'GTH_POTENTIALS'),
            pseudo_potential="GTH-PADE-q4",
        )
        fit_dict = calculator.analyse_structures(output_dict=result_dict)
        print(fit_dict)
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 66.44252286131331, atol=1e-04))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 72.38919826652857, atol=1e-04))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 4.453836551712821, atol=1e-04))
