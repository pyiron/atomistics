import os
import subprocess

from ase.build import bulk
import unittest

from atomistics.workflows import EnergyVolumeCurveWorkflow

try:
    from atomistics.calculators.sphinxdft import evaluate_with_sphinx

    skip_sphinx_test = False
except ImportError:
    skip_sphinx_test = True


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


def evaluate_sphinx(working_directory):
    output = subprocess.check_output(
        "sphinx", cwd=working_directory, shell=True, universal_newlines=True, env=os.environ.copy()
    )
    print(output)
    return output


@unittest.skipIf(
    skip_sphinx_test,
    "sphinxdft is not installed, so the sphinxdft based tests are skipped.",
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        workflow = EnergyVolumeCurveWorkflow(
            structure=bulk("Al", a=4.15, cubic=True),
            num_points=7,
            fit_type="polynomial",
            fit_order=3,
            vol_range=0.05,
            axes=("x", "y", "z"),
            strains=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_sphinx(
            task_dict=task_dict,
            working_directory=os.path.abspath("."),
            executable_function=evaluate_sphinx,
            maxSteps=100,
            eCut=25,
            kpoint_coords=[0.5, 0.5, 0.5],
            kpoint_folding=[4, 4, 4],
        )
        fit_dict = workflow.analyse_structures(output_dict=result_dict)
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))
