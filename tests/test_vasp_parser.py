import os
import shutil
from unittest import TestCase

from ase.build import bulk

from atomistics.calculators.vasp import (
    optimize_positions_with_vasp,
    optimize_positions_and_volume_with_vasp,
    calc_static_with_vasp,
    evaluate_with_vasp,
)
from atomistics.workflows import optimize_positions


os.environ["VASP_PP_PATH"] = os.path.abspath(os.path.join(__file__, "..", "static", "vasp"))

def copy_file_funct(working_directory):
    outcar_file = os.path.abspath(os.path.join(__file__, "..", "static", "vasp", "OUTCAR"))
    shutil.copy(outcar_file, working_directory)


class TestVaspParser(TestCase):
    def setUp(self):
        self._structure = bulk("Al", a=4.05, cubic=True)

    def test_optimize_positions_with_vasp(self):
        result = evaluate_with_vasp(
            task_dict=optimize_positions(self._structure),
            working_directory="opt_pos_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 1)