from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from phonopy.units import VaspToTHz
import unittest

from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import optimize_volume, PhonopyWorkflow


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
        workflow = PhonopyWorkflow(
            structure=result_dict["structure_with_optimized_volume"],
            interaction_range=10,
            factor=VaspToTHz,
            displacement=0.01,
            dos_mesh=20,
            primitive_matrix=None,
            number_of_snapshots=None,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_ase(task_dict=task_dict, ase_calculator=EMT())
        phonopy_dict = workflow.analyse_structures(output_dict=result_dict)
        mesh_dict, dos_dict = phonopy_dict["mesh_dict"], phonopy_dict["total_dos_dict"]
        self.assertEqual((324, 324), workflow.get_hesse_matrix().shape)
        self.assertTrue("qpoints" in mesh_dict.keys())
        self.assertTrue("weights" in mesh_dict.keys())
        self.assertTrue("frequencies" in mesh_dict.keys())
        self.assertTrue("eigenvectors" in mesh_dict.keys())
        self.assertTrue("group_velocities" in mesh_dict.keys())
        self.assertTrue("frequency_points" in dos_dict.keys())
        self.assertTrue("total_dos" in dos_dict.keys())
        dynmat_shape = workflow.get_dynamical_matrix().shape
        self.assertEqual(dynmat_shape[0], 12)
        self.assertEqual(dynmat_shape[1], 12)
        hessmat_shape = workflow.get_hesse_matrix().shape
        self.assertEqual(hessmat_shape[0], 324)
        self.assertEqual(hessmat_shape[1], 324)
        band_dict = workflow.get_band_structure()
        self.assertEqual(len(band_dict['qpoints']), 6)
        for vec in band_dict['qpoints']:
            self.assertTrue(vec.shape[0] in [34, 39, 78, 101, 95])
            self.assertEqual(vec.shape[1], 3)
        self.assertEqual(len(band_dict['distances']), 6)
        for vec in band_dict['distances']:
            self.assertTrue(vec.shape[0] in [34, 39, 78, 101, 95])
        self.assertEqual(len(band_dict['frequencies']), 6)
        for vec in band_dict['frequencies']:
            self.assertTrue(vec.shape[0] in [34, 39, 78, 101, 95])
            self.assertEqual(vec.shape[1], 12)
        self.assertIsNone(band_dict['eigenvectors'])
        self.assertIsNone(band_dict['group_velocities'])