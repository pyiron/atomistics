import os

from ase.build import bulk
import unittest

from atomistics.workflows import (
    analyse_results_for_energy_volume_curve,
    get_tasks_for_energy_volume_curve,
)

try:
    from atomistics.calculators import evaluate_with_qe

    skip_quantum_espresso_test = False
except ImportError:
    skip_quantum_espresso_test = True


def validate_fitdict(fit_dict):
    lst = [
        fit_dict["bulkmodul_eq"] > 50,
        fit_dict["bulkmodul_eq"] < 80,
        fit_dict["energy_eq"] > -2148.2,
        fit_dict["energy_eq"] < -2148.1,
        fit_dict["volume_eq"] > 70,
        fit_dict["volume_eq"] < 72,
    ]
    if not all(lst):
        print(fit_dict)
    return lst


@unittest.skipIf(
    skip_quantum_espresso_test,
    "pwtools is not installed, so the pwtools based quantum espresso tests are skipped.",
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        pseudopotentials = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"}
        task_dict = get_tasks_for_energy_volume_curve(
            structure=bulk("Al", a=4.15, cubic=True),
            num_points=7,
            vol_range=0.05,
            axes=("x", "y", "z"),
        )
        result_dict = evaluate_with_qe(
            task_dict=task_dict,
            calculation_name="espresso",
            working_directory=os.path.abspath("."),
            kpts=(3, 3, 3),
            pseudopotentials=pseudopotentials,
            tstress=True,
            tprnfor=True,
        )
        fit_dict = analyse_results_for_energy_volume_curve(
            output_dict=result_dict,
            task_dict=task_dict,
            fit_type="polynomial",
            fit_order=3,
        )
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))
