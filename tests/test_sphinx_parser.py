import os
import unittest
import shutil

import numpy as np
from ase.build import bulk

try:
    from sphinx_parser.toolkit import to_sphinx
    from atomistics.calculators.sphinxdft import (
        OutputParser,
        _generate_input,
        calc_static_with_sphinxdft,
        optimize_positions_with_sphinxdft,
        evaluate_with_sphinx,
        HARTREE_TO_EV,
        HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM,
        BOHR_TO_ANGSTROM,
    )

    skip_sphinx_test = False
except ImportError:
    skip_sphinx_test = True


def executable_function(working_directory):
    return "done in: " + working_directory


@unittest.skipIf(
    skip_sphinx_test,
    "sphinxdft is not installed, so the sphinxdft based tests are skipped.",
)
class TestSphinxParser(unittest.TestCase):
    def setUp(self):
        self._output_directory = os.path.abspath(os.path.join(__file__, "..", "static", "sphinxdft"))
        self._structure = bulk("Al", a=4.11, cubic=True)

    def test_output_parser(self):
        op = OutputParser(structure=self._structure, working_directory=self._output_directory)
        self.assertEqual(op.get_energy() / HARTREE_TO_EV, -8.369251265371 )
        self.assertEqual(op.get_volume(), 69.42653100000001)
        self.assertTrue(np.all(np.isclose(op.get_forces(), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))))
        with self.assertRaises(NotImplementedError):
            op.get_stress()

    def test_write_input(self):
        folder = "test"
        os.makedirs(folder, exist_ok=True)
        input_sx = to_sphinx(_generate_input(
            structure=self._structure,
            maxSteps=100,
            energy_cutoff_in_eV=25.0 * HARTREE_TO_EV,
            kpoint_coords=None,
            kpoint_folding=None,
        ))

        self.assertEqual(
            "	eCut = " + str(25.0 * HARTREE_TO_EV / HARTREE_TO_EV) + ";",
            "".join([l for l in input_sx.split("\n") if "eCut" in l])
        )
        self.assertEqual(
            "	folding = [3, 3, 3];",
            "".join([l for l in input_sx.split("\n") if "folding" in l])
        )
        self.assertEqual(
            "		potType = \"AtomPAW\";",
            "".join([l for l in input_sx.split("\n") if "potType" in l])
        )
        shutil.rmtree(folder)

    def test_calc_static(self):
        results = calc_static_with_sphinxdft(
            structure=self._structure,
            working_directory=self._output_directory,
            executable_function=executable_function,
            max_electronic_steps=100,
            energy_cutoff_in_eV=25.0 * HARTREE_TO_EV,
            kpoint_coords=None,
            kpoint_folding=None,
            output_keys=["volume", "forces", "energy"],
        )
        self.assertEqual(results["energy"] / HARTREE_TO_EV, -8.369251265371)
        self.assertEqual(results["volume"], 69.42653100000001)
        self.assertTrue(np.all(np.isclose(
            results["forces"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )))

    def test_optimize_positions_with_sphinxdft(self):
        structure_result = optimize_positions_with_sphinxdft(
            structure=self._structure,
            working_directory=self._output_directory,
            executable_function=executable_function,
            max_electronic_steps=100,
            energy_cutoff_in_eV=25.0 * HARTREE_TO_EV,
            kpoint_coords=None,
            kpoint_folding=None,
            mode="linQN",
            dEnergy=1.0e-6,
            max_ionic_steps=50,
        )
        self.assertEqual(len(structure_result), len(self._structure))

    def test_evaluate_with_sphinx_energy_and_forces(self):
        results = evaluate_with_sphinx(
            task_dict={"calc_energy": self._structure, "calc_forces": self._structure},
            working_directory=self._output_directory,
            executable_function=executable_function,
            max_electronic_steps=100,
            energy_cutoff_in_eV=25.0 * HARTREE_TO_EV,
            kpoint_coords=None,
            kpoint_folding=None,
        )
        self.assertEqual(results["energy"] / HARTREE_TO_EV, -8.369251265371)
        self.assertTrue(np.all(np.isclose(
            results["forces"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )))

    def test_evaluate_with_sphinx_structure_optimization(self):
        results = evaluate_with_sphinx(
            task_dict={"optimize_positions": self._structure},
            working_directory=self._output_directory,
            executable_function=executable_function,
            max_electronic_steps=100,
            energy_cutoff_in_eV=25.0 * HARTREE_TO_EV,
            kpoint_coords=None,
            kpoint_folding=None,
            sphinx_optimizer_kwargs={
                "mode": "linQN",
                "dEnergy": 1.0e-6,
                "max_ionic_steps": 50,
            },
        )
        self.assertTrue("structure_with_optimized_positions" in results.keys())
        self.assertEqual(len(results), 1)
