from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import QuasiHarmonicWorkflow, optimize_positions_and_volume


class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        structure = bulk("Al", cubic=True)
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT(),
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.000001}
        )
        workflow = QuasiHarmonicWorkflow(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            num_points=11,
            vol_range=0.05,
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=EMT()
        )
        eng_internal_dict, mesh_collect_dict, dos_collect_dict = workflow.analyse_structures(output_dict=result_dict)
        tp_collect_dict = workflow.get_thermal_properties(t_min=1, t_max=1500, t_step=50, temperatures=None)
        temperatures_qh, volumes_qh = workflow.get_thermal_expansion(output_dict=result_dict, temperatures=[100, 1000])
        self.assertEqual(len(eng_internal_dict.keys()), 11)
        self.assertEqual(len(tp_collect_dict.keys()), 11)
        self.assertEqual(len(temperatures_qh), 2)
        self.assertEqual(len(volumes_qh), 2)
        self.assertTrue(volumes_qh[0] < volumes_qh[-1])