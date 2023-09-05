import unittest
import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from atomistics.evcurve.calculator import EnergyVolumeCurveCalculator


def get_potential_energy_from_emt(structure):
    structure.calc = EMT()
    return structure.get_potential_energy()


class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        calculator = EnergyVolumeCurveCalculator(
            structure=bulk("Al", a=4.0, cubic=True),
            num_points=11,
            fit_type='polynomial',
            fit_order=3,
            vol_range=0.05,
            axes=['x', 'y', 'z'],
            strains=None,
        )
        structure_dict = calculator.generate_structures()
        energy_dict = {
            k: get_potential_energy_from_emt(structure=v)
            for k, v in structure_dict.items()
        }
        fit_dict = calculator.analyse_structures(output_dict=energy_dict)
        self.assertTrue(np.isclose(fit_dict['volume_eq'], 63.72615218844302))
        self.assertTrue(np.isclose(fit_dict['bulkmodul_eq'], 39.544084907317895))
        self.assertTrue(np.isclose(fit_dict['b_prime_eq'], 2.2509394023322566))
