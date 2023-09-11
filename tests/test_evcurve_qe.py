import shutil

from ase.build import bulk
import unittest

from atomistics.calculators.quantumespresso_ase.calculator import evaluate_with_quantumespresso
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow


quantum_espresso_command = "pw.x"
if shutil.which(quantum_espresso_command) is not None:
    skip_quantum_espresso_test = False
else:
    skip_quantum_espresso_test = True


def validate_fitdict(fit_dict):
    lst = [
        fit_dict['b_prime_eq'] > 3.0,
        fit_dict['b_prime_eq'] < 9.0,
        fit_dict['bulkmodul_eq'] > 52,
        fit_dict['bulkmodul_eq'] < 60,
        fit_dict['energy_eq'] > -2148.2,
        fit_dict['energy_eq'] < -2148.1,
        fit_dict['volume_eq'] > 70,
        fit_dict['volume_eq'] < 71,
    ]
    if not all(lst):
        print(fit_dict)
    return lst


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
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))
