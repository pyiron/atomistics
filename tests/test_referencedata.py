import unittest
from atomistics.referencedata import (
    get_chemical_information_from_mendeleev,
    get_chemical_information_from_wolframalpha,
    get_experimental_elastic_property_wikipedia,
)


class TestReferenceData(unittest.TestCase):
    def test_get_chemical_information_from_mendeleev(self):
        al_data = get_chemical_information_from_mendeleev(chemical_symbol="Al")
        self.assertEqual(al_data["atomic_number"], 13)

    def test_get_chemical_information_from_wolframalpha(self):
        al_data = get_chemical_information_from_wolframalpha(chemical_element="Al")
        self.assertEqual(al_data["latticeconstant"], "(404.95, 404.95, 404.95)")

    def test_get_experimental_elastic_property_wikipedia(self):
        al_data = get_experimental_elastic_property_wikipedia(chemical_symbol="Al")
        self.assertEqual(
            al_data,
            {'bulk_modulus': 76.0, 'poissons_ratio': 0.35, 'shear_modulus': 26.0, 'youngs_modulus': 70.0}
        )