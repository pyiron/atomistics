import shutil

from ase.build import bulk
import numpy as np
import unittest

from atomistics.calculators.quantumespresso_ase.calculator import evaluate_with_quantumespresso
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow


quantum_espresso_command = "pw.x"
if shutil.which(quantum_espresso_command) is not None:
    skip_quantum_espresso_test = False
else:
    skip_quantum_espresso_test = True


@unittest.skipIf(
    skip_quantum_espresso_test, "quantum_espresso is not installed, so the quantum_espresso tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        pseudopotentials = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"}
        calculator = EnergyVolumeCurveWorkflow(
            structure=bulk("Al", a=4.15, cubic=True),
            num_points=11,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=['x', 'y', 'z'],
            strains=None,
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_quantumespresso(
            task_dict=structure_dict,
            pseudopotentials=pseudopotentials,
            tstress=True,
            tprnfor=True,
            kpts=(3, 3, 3),
        )
        fit_dict = calculator.analyse_structures(output_dict=result_dict)
        print(fit_dict)
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 70.86657718794973, atol=1e-04))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 55.28802217387117, atol=1e-04))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 5.3705786417858805, atol=1e-04))
