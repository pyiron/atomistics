from ase.build import bulk
import numpy as np

import unittest

from atomistics.workflows.evcurve.fit import fit_equation_of_state
from atomistics.workflows.evcurve.debye import DebyeModel
from atomistics.workflows.evcurve.thermo import get_thermo_bulk_model


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
        print(fit_dict)
        self.assertAlmostEqual(fit_dict["volume_eq"], 66.43032532814925)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 77.61265363975706)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 1.2734991618131122)


class TestThermo(unittest.TestCase):
    def setUp(self):
        self._volumes = [
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
        self._thermo = get_thermo_bulk_model(
            temperatures=np.arange(1.0, 1550.0, 50.0),
            debye_model=DebyeModel(
                fit_dict={
                    "energy_eq": -13.439961346483864,
                    "volume_eq": 66.43032532814925,
                    "bulkmodul_eq": 77.61265363975706,
                    "b_prime_eq": 1.2734991618131122,
                    "volume": self._volumes,
                    "fit_dict": {
                        "fit_type": "vinet",
                    }
                },
                masses=bulk("Al", cubic=True).get_masses(),
                num_steps=50,
            ),
        )

    def test_copy(self):
        thermo_copy = self._thermo.copy()
        self.assertEqual(len(thermo_copy.__dict__), len(self._thermo.__dict__))

    def test_num_atoms(self):
        self.assertEqual(self._thermo.num_atoms, 1)
        self._thermo.num_atoms = 2
        self.assertEqual(self._thermo.num_atoms, 2)
        self._thermo.num_atoms = 1

    def test_d_temp(self):
        self.assertEqual(self._thermo._d_temp, 50.0)

    def test_d_vol(self):
        self.assertTrue(np.isclose(self._thermo._d_vol, 0.13557215186849447))

    def test_volumes(self):
        self.assertTrue(np.isclose(self._thermo.volumes[0], self._volumes[0]))
        self.assertTrue(np.isclose(self._thermo.volumes[-1], self._volumes[-1]))
        self.assertEqual(len(self._thermo.volumes), 50)

    def test_entropy(self):
        self._thermo._entropy = None
        self.assertEqual(self._thermo.entropy.shape[0], 31)
        self.assertEqual(self._thermo.entropy.shape[1], 50)
        self._thermo._entropy = None

    def test_pressure(self):
        self._thermo._pressure = None
        self.assertEqual(self._thermo.pressure.shape[0], 31)
        self.assertEqual(self._thermo.pressure.shape[1], 50)
        self._thermo._pressure = None
