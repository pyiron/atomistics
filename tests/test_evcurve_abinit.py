import shutil

from ase.build import bulk
from ase.units import Ry
import unittest

from atomistics.calculators.abinit_ase.calculator import evaluate_with_abinit
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow


if shutil.which("abinit") is not None:
    skip_abinit_test = False
else:
    skip_abinit_test = True


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
    skip_abinit_test, "abinit is not installed, so the abinit tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
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
        result_dict = evaluate_with_abinit(
            task_dict=structure_dict,
            label='abinit_evcurve',
            nbands=32,
            ecut=10 * Ry,
            kpts=(3, 3, 3),
            toldfe=1.0e-2,
            v8_legacy_format=False,
        )
        fit_dict = calculator.analyse_structures(output_dict=result_dict)
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))
