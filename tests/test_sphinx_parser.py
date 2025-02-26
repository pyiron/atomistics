import os
import unittest
import shutil

import numpy as np
from ase.build import bulk

try:
    from atomistics.calculators.sphinxdft import OutputParser, _write_input, calc_static_with_sphinxdft, evaluate_with_sphinx

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
        self.assertEqual(op.get_energy(), -227.73892877167552)
        self.assertEqual(op.get_volume(), 69.42653100000001)
        self.assertTrue(np.all(np.isclose(op.get_forces(), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))))
        with self.assertRaises(NotImplementedError):
            op.get_stress()

    def test_write_input(self):
        folder = "test"
        os.makedirs(folder, exist_ok=True)
        _write_input(
            structure=self._structure,
            working_directory=folder,
            maxSteps=100,
            eCut=25.0,
            kpoint_coords=None,
            kpoint_folding=None,
        )
        file = os.path.join(folder, "input.sx")
        self.assertTrue(os.path.exists(file))
        with open(file, "r") as f:
            lines = f.readlines()

        self.assertEqual(
            "	cell = [[7.766774377480988, 0.0, 0.0], [0.0, 7.766774377480988, 0.0], [0.0, 0.0, 7.766774377480988]];\n",
            "".join([l for l in lines if "cell" in l])
        )
        self.assertEqual(
            "	eCut = 25.0;\n",
            "".join([l for l in lines if "eCut" in l])
        )
        self.assertEqual(
            "	folding = [3, 3, 3];\n",
            "".join([l for l in lines if "folding" in l])
        )
        self.assertEqual(
            "		potType = \"AtomPAW\";\n",
            "".join([l for l in lines if "potType" in l])
        )
        shutil.rmtree(folder)

    def test_calc_static(self):
        results = calc_static_with_sphinxdft(
            structure=self._structure,
            working_directory=self._output_directory,
            executable_function=executable_function,
            maxSteps=100,
            eCut=25,
            kpoint_coords=None,
            kpoint_folding=None,
            output_keys=["volume", "forces", "energy"],
        )
        self.assertEqual(results["energy"], -227.73892877167552)
        self.assertEqual(results["volume"], 69.42653100000001)
        self.assertTrue(np.all(np.isclose(
            results["forces"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )))

    def test_evaluate_with_sphinx(self):
        results = evaluate_with_sphinx(
            task_dict={"calc_energy": self._structure, "calc_forces": self._structure},
            working_directory=self._output_directory,
            executable_function=executable_function,
            maxSteps=100,
            eCut=25,
            kpoint_coords=None,
            kpoint_folding=None,
        )
        self.assertEqual(results["energy"], -227.73892877167552)
        self.assertTrue(np.all(np.isclose(
            results["forces"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )))
