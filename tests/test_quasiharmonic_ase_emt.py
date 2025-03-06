from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import QuasiHarmonicWorkflow, optimize_volume


class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        structure = bulk("Al", cubic=True)
        task_dict = optimize_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
        )
        workflow = QuasiHarmonicWorkflow(
            structure=result_dict["structure_with_optimized_volume"],
            num_points=11,
            vol_range=0.10,
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(task_dict=task_dict, ase_calculator=EMT())
        eng_internal_dict, phonopy_collect_dict = workflow.analyse_structures(
            output_dict=result_dict
        )
        tp_collect_dict = workflow.get_thermal_properties(
            t_min=1, t_max=1500, t_step=50, temperatures=None
        )
        thermal_properties_dict = workflow.get_thermal_properties(
            temperatures=[100, 1000],
            output_keys=["free_energy", "temperatures", "volumes"],
            quantum_mechanical=True,
        )
        temperatures_qh_qm, volumes_qh_qm = (
            thermal_properties_dict["temperatures"],
            thermal_properties_dict["volumes"],
        )
        thermal_properties_dict = workflow.get_thermal_properties(
            temperatures=[100, 1000],
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
