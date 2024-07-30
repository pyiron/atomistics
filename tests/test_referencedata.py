import unittest
from atomistics.referencedata import get_elastic_properties_from_wikipedia


try:
    from atomistics.referencedata import (
        get_chemical_information_from_mendeleev,
        get_chemical_information_from_wolframalpha,
    )

    mendeleev_not_available = False
except ImportError:
    mendeleev_not_available = True


try:
    import lxml

    lxml_not_available = False
except ImportError:
    lxml_not_available = True


class TestReferenceData(unittest.TestCase):
    @unittest.skipIf(
        mendeleev_not_available,
        "mendeleev is not installed, so the mendeleev tests are skipped.",
    )
    def test_get_chemical_information_from_mendeleev(self):
        al_data = get_chemical_information_from_mendeleev(chemical_symbol="Al")
        self.assertEqual(al_data["atomic_number"], 13)

    @unittest.skipIf(
        mendeleev_not_available,
        "mendeleev is not installed, so the mendeleev tests are skipped.",
    )
    def test_get_chemical_information_from_wolframalpha(self):
        al_data = get_chemical_information_from_wolframalpha(chemical_symbol="Al")
        self.assertEqual(al_data["latticeconstant"], "(404.95, 404.95, 404.95)")

    @unittest.skipIf(
        lxml_not_available, "lxml is not installed, so the lxml tests are skipped."
    )
    def test_get_experimental_elastic_property_wikipedia(self):
        al_data = get_elastic_properties_from_wikipedia(chemical_symbol="Al")
        self.assertEqual(
            al_data,
            {
                "bulk_modulus": 76.0,
                "poissons_ratio": 0.35,
                "shear_modulus": 26.0,
                "youngs_modulus": 70.0,
            },
        )
