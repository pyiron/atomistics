import os
import shutil

from ase.build import bulk
import unittest

from atomistics.calculators import evaluate_with_qe
from atomistics.workflows import EnergyVolumeCurveWorkflow, optimize_positions_and_volume


quantum_espresso_command = "pw.x"
if shutil.which(quantum_espresso_command) is not None:
    skip_quantum_espresso_test = False
else:
    skip_quantum_espresso_test = True


def validate_fitdict(fit_dict):
    lst = [
        fit_dict['bulkmodul_eq'] > 50,
        fit_dict['bulkmodul_eq'] < 80,
        fit_dict['energy_eq'] > -2148.2,
        fit_dict['energy_eq'] < -2148.1,
        fit_dict['volume_eq'] > 71,
        fit_dict['volume_eq'] < 72,
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
        structure = bulk("Al", cubic=True)
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_qe(
            task_dict=task_dict,
            calculation_name="espresso",
            working_directory=os.path.abspath("."),
            kpts=(3, 3, 3),
            pseudopotentials=pseudopotentials,
            tstress=True,
            tprnfor=True,
        )
        workflow = EnergyVolumeCurveWorkflow(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            num_points=5,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=('x', 'y', 'z'),
            strains=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_qe(
            task_dict=task_dict,
            calculation_name="espresso",
            working_directory=os.path.abspath("."),
            kpts=(3, 3, 3),
            pseudopotentials=pseudopotentials,
            tstress=True,
            tprnfor=True,
        )
        fit_dict = workflow.analyse_structures(output_dict=result_dict)
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))
