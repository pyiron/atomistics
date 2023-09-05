import unittest
import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from atomistics.elastic.calculator import ElasticMatrixCalculator


def get_potential_energy_from_emt(structure):
    structure.calc = EMT()
    return structure.get_potential_energy()


class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        calculator = calculator = ElasticMatrixCalculator(
            basis_ref=bulk("Al", a=4.0, cubic=True),
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
            fit_order=2
        )
        structure_dict = calculator.generate_structures()
        energy_dict = {
            k: get_potential_energy_from_emt(structure=v)
            for k, v in structure_dict.items()
        }
        elastic_dict = calculator.analyse_structures(output_dict=energy_dict)
        self.assertTrue(np.isclose(elastic_dict["C"][0, 0], 52.62435421))
        self.assertTrue(np.isclose(elastic_dict["C"][0, 1], 32.6743838))
        self.assertTrue(np.isclose(elastic_dict["C"][3, 3], 35.58677436))
