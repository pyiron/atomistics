from ase.build import bulk
from ase.optimize import LBFGS
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import (
    get_tasks_for_quasi_harmonic_approximation,
    get_thermal_properties_for_quasi_harmonic_approximation,
    analyse_results_for_quasi_harmonic_approximation,
    optimize_positions_and_volume,
)


try:
    from mace.calculators import mace_mp

    skip_mace_test = False
except ImportError:
    skip_mace_test = True


@unittest.skipIf(
    skip_mace_test, "mace is not installed, so the mace tests are skipped."
)
class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        structure = bulk("Al", cubic=True)
        ase_calculator = mace_mp(
            model="medium", dispersion=False, default_dtype="float32", device="cpu"
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=ase_calculator,
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.001},
        )
        task_dict, qh_dict = get_tasks_for_quasi_harmonic_approximation(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            num_points=11,
            vol_range=0.10,
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=ase_calculator,
        )
        eng_internal_dict, phonopy_collect_dict = analyse_results_for_quasi_harmonic_approximation(
            output_dict=result_dict,
            qh_dict=qh_dict,
        )
        tp_collect_dict = get_thermal_properties_for_quasi_harmonic_approximation(
            eng_internal_dict=eng_internal_dict,
            task_dict=task_dict,
            qh_dict=qh_dict,
            fit_type="polynomial",
            fit_order=3,
            t_min=1,
            t_max=501,
            t_step=50,
            temperatures=None
        )
        thermal_properties_dict = get_thermal_properties_for_quasi_harmonic_approximation(
            eng_internal_dict=eng_internal_dict,
            task_dict=task_dict,
            qh_dict=qh_dict,
            fit_type="polynomial",
            fit_order=3,
            temperatures=[100, 500],
            output_keys=["free_energy", "temperatures", "volumes"],
            quantum_mechanical=True,
        )
        temperatures_qh_qm, volumes_qh_qm = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        thermal_properties_dict = get_thermal_properties_for_quasi_harmonic_approximation(
            eng_internal_dict=eng_internal_dict,
            task_dict=task_dict,
            qh_dict=qh_dict,
            fit_type="polynomial",
            fit_order=3,
            temperatures=[100, 500],
            output_keys=["free_energy", "temperatures", "volumes"],
            quantum_mechanical=False,
        )
        temperatures_qh_cl, volumes_qh_cl = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        self.assertEqual(len(eng_internal_dict.keys()), 11)
        self.assertEqual(len(tp_collect_dict.keys()), 5)
        self.assertEqual(len(temperatures_qh_qm), 2)
        self.assertEqual(len(volumes_qh_qm), 2)
        self.assertTrue(volumes_qh_qm[0] < volumes_qh_qm[-1])
        self.assertEqual(len(temperatures_qh_cl), 2)
        self.assertEqual(len(volumes_qh_cl), 2)
        self.assertTrue(volumes_qh_cl[0] < volumes_qh_cl[-1])
