from ase import units
from ase.build import bulk
from atomistics.calculators import calc_molecular_dynamics_langevin_with_ase
import unittest


try:
    import matgl
    from matgl.ext.ase import M3GNetCalculator

    skip_matgl_test = False
except ImportError:
    skip_matgl_test = True


@unittest.skipIf(
    skip_matgl_test, "matgl is not installed, so the matgl tests are skipped."
)
class TestASEMD(unittest.TestCase):
    def test_ase_langevin(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        ase_calculator = M3GNetCalculator(matgl.load_model("M3GNet-MP-2021.2.8-PES"))
        result_dict = calc_molecular_dynamics_langevin_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            run=100,
            thermo=10,
            timestep=1 * units.fs,
            temperature=100,
            friction=0.002,
        )
        self.assertEqual(result_dict["positions"].shape, (10, 32, 3))
        self.assertEqual(result_dict["velocities"].shape, (10, 32, 3))
        self.assertEqual(result_dict["cell"].shape, (10, 3, 3))
        self.assertEqual(result_dict["forces"].shape, (10, 32, 3))
        self.assertEqual(result_dict["temperature"].shape, (10, ))
        self.assertEqual(result_dict["energy_pot"].shape, (10, ))
        self.assertEqual(result_dict["energy_tot"].shape, (10, ))
        self.assertEqual(result_dict["pressure"].shape, (10, 3, 3))
        self.assertTrue(result_dict["temperature"][-1] > 25)
        self.assertTrue(result_dict["temperature"][-1] < 75)