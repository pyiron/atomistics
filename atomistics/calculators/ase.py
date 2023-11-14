from __future__ import annotations

from typing import TYPE_CHECKING

from atomistics.calculators.wrapper import as_task_dict_evaluator

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator as ASECalculator
    from ase.optimize.optimize import Optimizer
    from atomistics.calculators.interface import TaskName


def ase_optimize_structure(
    structure, ase_calculator, ase_optimizer, ase_optimizer_kwargs
):
    structure_optimized = structure.copy()
    structure_optimized.calc = ase_calculator
    ase_optimizer_obj = ase_optimizer(structure_optimized)
    ase_optimizer_obj.run(**ase_optimizer_kwargs)
    return structure_optimized


@as_task_dict_evaluator
def evaluate_with_ase(
    structure: Atoms,
    tasks: list[TaskName],
    ase_calculator: ASECalculator,
    ase_optimizer: Optimizer = None,
    ase_optimizer_kwargs: dict = None,
):
    results = {}
    if "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = ase_optimize_structure(
            structure=structure,
            ase_calculator=ase_calculator,
            ase_optimizer=ase_optimizer,
            ase_optimizer_kwargs=ase_optimizer_kwargs,
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks:
        structure.calc = ase_calculator
        if "calc_energy" in tasks:
            results["energy"] = structure.get_potential_energy()
        if "calc_forces" in tasks:
            results["forces"] = structure.get_forces()
    else:
        raise ValueError("The ASE calculator does not implement:", tasks)
    return results
