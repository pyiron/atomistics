import numpy as np

import unittest

from atomistics.workflows.evcurve.fit import fit_equation_of_state


class TestEvCurve(unittest.TestCase):
    def setUp(self):
        self.volumes = [
            63.10883669478296,
            63.77314023893856,
            64.43744378309412,
            65.10174732724975,
            65.7660508714054,
            66.43035441556098,
            67.09465795971657,
            67.7589615038722,
            68.42326504802779,
            69.08756859218344,
            69.75187213633905,
        ]
        self.energies = [
            -13.39817505470619,
            -13.4133940159381,
            -13.425115937672247,
            -13.433413658516752,
            -13.438358754759532,
            -13.439999952735112,
            -13.438382355644501,
            -13.433605756604651,
            -13.42577121684493,
            -13.41495739484744,
            -13.401227593921211,
        ]

    def test_birch(self):
        fit_dict = fit_equation_of_state(
            volume_lst=self.volumes, energy_lst=self.energies, fittype="birch"
        )
        self.assertAlmostEqual(fit_dict["volume_eq"], 66.43027009811671)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 77.7433780646763)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 1.2836228593874182)

    def test_birchmurnaghan(self):
        fit_dict = fit_equation_of_state(
            volume_lst=self.volumes, energy_lst=self.energies, fittype="birchmurnaghan"
        )
        self.assertAlmostEqual(fit_dict["volume_eq"], 66.43027009811708)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 77.74337806467966)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 1.2836228593684815)

    def test_murnaghan(self):
        fit_dict = fit_equation_of_state(
            volume_lst=self.volumes, energy_lst=self.energies, fittype="murnaghan"
        )
        self.assertAlmostEqual(fit_dict["volume_eq"], 66.43035753542675)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 77.60443933015738)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 1.2716548170090776)

    def test_pouriertarantola(self):
        fit_dict = fit_equation_of_state(
            volume_lst=self.volumes,
            energy_lst=self.energies,
            fittype="pouriertarantola",
        )
        self.assertAlmostEqual(fit_dict["volume_eq"], 66.43035598678892)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 77.61743376692809)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 1.272111993713677)

    def test_vinet(self):
        fit_dict = fit_equation_of_state(
            volume_lst=self.volumes, energy_lst=self.energies, fittype="vinet"
        )
        self.assertAlmostEqual(fit_dict["volume_eq"], 66.43032532814925)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 77.61265363975706)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 1.2734991618131122)
