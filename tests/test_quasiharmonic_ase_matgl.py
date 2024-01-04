from ase.build import bulk
from ase.optimize import LBFGS
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import QuasiHarmonicWorkflow, optimize_positions_and_volume


try:
    import matgl
    from matgl.ext.ase import M3GNetCalculator

    skip_matgl_test = False
except ImportError:
    skip_matgl_test = True


@unittest.skipIf(
    skip_matgl_test, "matgl is not installed, so the matgl tests are skipped."
)
class TestPhonons(unittest.TestCase):
    def test_calc_phonons(self):
        structure = bulk("Al", cubic=True)
        ase_calculator = M3GNetCalculator(matgl.load_model("M3GNet-MP-2021.2.8-PES"))
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=ase_calculator,
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.001}
        )
        workflow = QuasiHarmonicWorkflow(
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
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=ase_calculator,
        )
        eng_internal_dict, phonopy_collect_dict = workflow.analyse_structures(output_dict=result_dict)
        tp_collect_dict = workflow.get_thermal_properties(t_min=1, t_max=501, t_step=50, temperatures=None)
        temperatures_qh_qm, volumes_qh_qm = workflow.get_thermal_expansion(
            output_dict=result_dict,
            temperatures=[100, 500],
            quantum_mechanical=True
        )
        temperatures_qh_cl, volumes_qh_cl = workflow.get_thermal_expansion(
            output_dict=result_dict,
            temperatures=[100, 500],
            quantum_mechanical=False
        )
        self.assertEqual(len(eng_internal_dict.keys()), 11)
        self.assertEqual(len(tp_collect_dict.keys()), 5)
        self.assertEqual(len(temperatures_qh_qm), 2)
        self.assertEqual(len(volumes_qh_qm), 2)
        self.assertTrue(volumes_qh_qm[0] < volumes_qh_qm[-1])
        self.assertEqual(len(temperatures_qh_cl), 2)
        self.assertEqual(len(volumes_qh_cl), 2)
        self.assertTrue(volumes_qh_cl[0] < volumes_qh_cl[-1])
