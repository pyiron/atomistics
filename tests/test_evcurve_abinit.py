import shutil

from ase.build import bulk
from ase.calculators.abinit import Abinit
from ase.units import Ry
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import EnergyVolumeCurveWorkflow


if shutil.which("abinit") is not None:
    skip_abinit_test = False
else:
    skip_abinit_test = True


def validate_fitdict(fit_dict):
    lst = [
        fit_dict['bulkmodul_eq'] > 130,
        fit_dict['bulkmodul_eq'] < 140,
        fit_dict['energy_eq'] > -227,
        fit_dict['energy_eq'] < -226,
        fit_dict['volume_eq'] > 67,
        fit_dict['volume_eq'] < 68,
    ]
    if not all(lst):
        print(fit_dict)
    return lst


@unittest.skipIf(
    skip_abinit_test, "abinit is not installed, so the abinit tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        workflow = EnergyVolumeCurveWorkflow(
            structure=bulk("Al", a=4.05, cubic=True),
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
            ase_calculator=Abinit(
                label='abinit_evcurve',
                nbands=32,
                ecut=10 * Ry,
                kpts=(3, 3, 3),
                toldfe=1.0e-2,
                v8_legacy_format=False,
            )
        )
        fit_dict = workflow.analyse_structures(output_dict=result_dict)
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))
