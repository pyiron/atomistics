import unittest

from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow

from .available_evaluators import AL_UNIT, Calculators, EVALUATION_FUNCTIONS


class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        workflow = EnergyVolumeCurveWorkflow(
            structure=AL_UNIT,
            num_points=11,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=['x', 'y', 'z'],
            strains=None,
        )
        structure_dict = workflow.generate_structures()

        for calculator, expected in [
            (Calculators.abinit, (67, 130, None)),  # WILL BE WRONG -- original tests a range and no B'
            (Calculators.emt, (63.726152188443, 39.5440849073178, 2.25093940233225)),
            (Calculators.gpaw, (66.442522861313, 72.389198266528, 4.4538365517128)),
            (Calculators.lammps, (66.430198531039, 77.72501359531, 1.2795024590799)),
            (Calculators.qe, (70, 52, 3.0)),  # WILL BE WRONG -- original tests a range and a=4.15
        ]:
            evaluate = EVALUATION_FUNCTIONS[calculator]
            if evaluate is not None:
                with self.subTest(f"Evaluating with {calculator}"):
                    result_dict = evaluate(structure_dict)
                    fit_dict = workflow.analyse_structures(output_dict=result_dict)
                    print(fit_dict)
                    self.assertAlmostEqual(fit_dict['volume_eq'], expected[0])
                    self.assertAlmostEqual(fit_dict['bulkmodul_eq'], expected[1])
                    self.assertAlmostEqual(fit_dict['b_prime_eq'], expected[2])
