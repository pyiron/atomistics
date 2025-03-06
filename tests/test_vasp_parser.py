import os
import shutil
from unittest import TestCase

from ase.build import bulk

from atomistics.calculators import (
    evaluate_with_vasp,
    optimize_cell_with_vasp,
    optimize_volume_with_vasp,
    optimize_positions_with_vasp,
    optimize_positions_and_volume_with_vasp,
)
from atomistics.workflows import optimize_positions, optimize_positions_and_volume, optimize_volume, optimize_cell


os.environ["VASP_PP_PATH"] = os.path.abspath(os.path.join(__file__, "..", "static", "vasp"))

def copy_file_funct(working_directory):
    outcar_file = os.path.abspath(os.path.join(__file__, "..", "static", "vasp", "OUTCAR"))
    shutil.copy(outcar_file, working_directory)


class TestVaspParser(TestCase):
    def setUp(self):
        self._structure = bulk("Al", a=4.05, cubic=True)

    def test_optimize_positions_with_vasp_task(self):
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

    def test_optimize_positions_with_vasp(self):
        result = optimize_positions_with_vasp(
            structure=self._structure,
            working_directory="opt_pos_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 4)

    def test_optimize_positions_and_volume_with_vasp_task(self):
        result = evaluate_with_vasp(
            task_dict=optimize_positions_and_volume(self._structure),
            working_directory="opt_pos_and_vol_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 1)

    def test_optimize_positions_and_volume_with_vasp(self):
        result = optimize_positions_and_volume_with_vasp(
            structure=self._structure,
            working_directory="opt_pos_and_vol_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 4)

    def test_optimize_cell_with_vasp_task(self):
        result = evaluate_with_vasp(
            task_dict=optimize_cell(self._structure),
            working_directory="opt_cell_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 1)

    def test_optimize_cell_with_vasp(self):
        result = optimize_cell_with_vasp(
            structure=self._structure,
            working_directory="opt_cell_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 4)

    def test_optimize_volume_with_vasp_task(self):
        result = evaluate_with_vasp(
            task_dict=optimize_volume(self._structure),
            working_directory="opt_vol_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 1)

    def test_optimize_volume_with_vasp(self):
        result = optimize_volume_with_vasp(
            structure=self._structure,
            working_directory="opt_vol_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 4)

    def test_calc_static_task(self):
        result = evaluate_with_vasp(
            task_dict={
                "calc_energy": self._structure,
                "calc_forces": self._structure,
                "calc_stress": self._structure,
            },
            working_directory="opt_vol_task",
            executable_function=copy_file_funct,
            prec="Accurate",
            algo="Fast",
            lreal=False,
            lwave=False,
            lorbit=0,
            xc="pbe"
        )
        self.assertEqual(len(result), 3)
