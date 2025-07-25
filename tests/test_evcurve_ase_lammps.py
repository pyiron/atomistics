from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.optimize import LBFGS
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import (
    analyse_results_for_energy_volume_curve,
    get_tasks_for_energy_volume_curve,
    get_thermal_properties_for_energy_volume_curve,
    optimize_volume,
)


try:
    from lammps import lammps

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the lammps tests are skipped."
)
class TestEvCurve(unittest.TestCase):
    def test_calc_evcurve(self):
        cmds = ["pair_style morse/smooth/linear 9.0", "pair_coeff * * 0.5 1.8 2.95"]
        structure = bulk("Al", cubic=True)
        task_dict = optimize_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=LAMMPSlib(
                lmpcmds=cmds,
                atom_types={"Al": 1},
                keep_alive=True,
            ),
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
        )
        task_dict = get_tasks_for_energy_volume_curve(
            structure=result_dict["structure_with_optimized_volume"],
            num_points=11,
            vol_range=0.05,
            axes=("x", "y", "z"),
        )
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=LAMMPSlib(
                lmpcmds=cmds,
                atom_types={"Al": 1},
                keep_alive=True,
            ),
        )
        fit_dict = analyse_results_for_energy_volume_curve(
            output_dict=result_dict,
            task_dict=task_dict,
            fit_type="polynomial",
            fit_order=3,
        )
        thermal_properties_dict = get_thermal_properties_for_energy_volume_curve(
            fit_dict=fit_dict,
            masses=structure.get_masses(),
            temperatures=[100, 1000],
            output_keys=["temperatures", "volumes"],
        )
        temperatures_ev, volumes_ev = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        self.assertAlmostEqual(fit_dict["volume_eq"], 66.29753110818122)
        self.assertAlmostEqual(fit_dict["bulkmodul_eq"], 218.25686471974936)
        self.assertAlmostEqual(fit_dict["b_prime_eq"], 6.218603542219656)
        self.assertEqual(len(temperatures_ev), 2)
        self.assertEqual(len(volumes_ev), 2)
        self.assertTrue(volumes_ev[0] < volumes_ev[-1])
