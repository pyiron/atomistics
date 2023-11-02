import os
import shutil

from ase.build import bulk
from ase.calculators.siesta import Siesta
from ase.units import Ry
import unittest

from atomistics.calculators.ase import evaluate_with_ase
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow


siesta_command = "siesta"
if shutil.which(siesta_command) is not None:
    skip_siesta_test = False
else:
    skip_siesta_test = True


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
    skip_siesta_test, "siesta is not installed, so the siesta tests are skipped."
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
        result_dict = evaluate_with_ase(
            task_dict=structure_dict,
            ase_calculator=Siesta(
                label="siesta",
                xc="PBE",
                mesh_cutoff=200 * Ry,
                energy_shift=0.01 * Ry,
                basis_set="DZ",
                kpts=(5, 5, 5),
                fdf_arguments={"DM.MixingWeight": 0.1, "MaxSCFIterations": 100},
                pseudo_path=os.path.abspath("static/siesta"),
                pseudo_qualifier="",
            )
        )
        fit_dict = calculator.analyse_structures(output_dict=result_dict)
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))
