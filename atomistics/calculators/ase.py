from __future__ import annotations

from typing import TYPE_CHECKING

from atomistics.calculators.wrapper import as_task_dict_evaluator

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator as ASECalculator
    from atomistics.calculators.interface import TaskName


@as_task_dict_evaluator
def evaluate_with_ase(
    structure: Atoms, tasks: list[TaskName], ase_calculator: ASECalculator
):
    structure.calc = ase_calculator
    results = {}
    if "calc_energy" in tasks:
        results["energy"] = structure.get_potential_energy()
    if "calc_forces" in tasks:
        results["forces"] = structure.get_forces()
    return results
