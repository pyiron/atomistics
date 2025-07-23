import os
import shutil

from ase.build import bulk
from ase.calculators.siesta import Siesta
from ase.units import Ry
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import (
    get_tasks_for_energy_volume_curve,
    analyse_results_for_energy_volume_curve,
)


siesta_command = "siesta"
if shutil.which(siesta_command) is not None:
    skip_siesta_test = False
else:
    skip_siesta_test = True


def validate_fitdict(fit_dict):
    lst = [
        fit_dict["b_prime_eq"] > 14.0,
        fit_dict["b_prime_eq"] < 15.0,
        fit_dict["bulkmodul_eq"] > 42,
        fit_dict["bulkmodul_eq"] < 44,
        fit_dict["energy_eq"] > -281.0,
        fit_dict["energy_eq"] < -280.9,
        fit_dict["volume_eq"] > 74,
        fit_dict["volume_eq"] < 75,
    ]
    if not all(lst):
        print(fit_dict)
    return lst


@unittest.skipIf(
    skip_siesta_test, "siesta is not installed, so the siesta tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        structure = bulk("Al", a=4.15, cubic=True)
        task_dict = get_tasks_for_energy_volume_curve(
            structure=structure,
            num_points=11,
            vol_range=0.05,
            axes=("x", "y", "z"),
        )
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=Siesta(
                label="siesta",
                xc="PBE",
                mesh_cutoff=200 * Ry,
                energy_shift=0.01 * Ry,
                basis_set="DZ",
                kpts=(5, 5, 5),
                fdf_arguments={"DM.MixingWeight": 0.1, "MaxSCFIterations": 100},
                pseudo_path=os.path.abspath("tests/static/siesta"),
                pseudo_qualifier="",
            ),
        )
        fit_dict = analyse_results_for_energy_volume_curve(
            output_dict=result_dict,
            task_dict=task_dict,
            fit_type="polynomial",
            fit_order=3,
        )
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))
