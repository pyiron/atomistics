from __future__ import annotations

from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import UnitCellFilter
import numpy as np
from typing import TYPE_CHECKING

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.shared.output import OutputStatic, OutputMolecularDynamics
from atomistics.calculators.wrapper import as_task_dict_evaluator

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from ase.calculators.calculator import Calculator as ASECalculator
    from ase.optimize.optimize import Optimizer
    from atomistics.calculators.interface import TaskName


class ASEExecutor(object):
    def __init__(self, ase_structure, ase_calculator):
        self.structure = ase_structure
        self.structure.calc = ase_calculator

    def get_forces(self):
        return self.structure.get_forces()

    def get_energy(self):
        return self.structure.get_potential_energy()

    def get_stress(self):
        return self.structure.get_stress(voigt=False)

    def get_total_energy(self):
        return (
            self.structure.get_potential_energy() + self.structure.get_kinetic_energy()
        )

    def get_cell(self):
        return self.structure.get_cell()

    def get_positions(self):
        return self.structure.get_positions()

    def get_velocities(self):
        return self.structure.get_velocities()

    def get_temperature(self):
        return self.structure.get_temperature()


ASEOutputStatic = OutputStatic(
    forces=ASEExecutor.get_forces,
    energy=ASEExecutor.get_energy,
    stress=ASEExecutor.get_stress,
)

ASEOutputMolecularDynamics = OutputMolecularDynamics(
    positions=ASEExecutor.get_positions,
    cell=ASEExecutor.get_cell,
    forces=ASEExecutor.get_forces,
    temperature=ASEExecutor.get_temperature,
    energy_pot=ASEExecutor.get_energy,
    energy_tot=ASEExecutor.get_total_energy,
    pressure=ASEExecutor.get_stress,
    velocities=ASEExecutor.get_velocities,
)


@as_task_dict_evaluator
def evaluate_with_ase(
    structure: Atoms,
    tasks: list[TaskName],
    ase_calculator: ASECalculator,
    ase_optimizer: Optimizer = None,
    ase_optimizer_kwargs: dict = {},
):
    results = {}
    if "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = optimize_positions_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            ase_optimizer=ase_optimizer,
            ase_optimizer_kwargs=ase_optimizer_kwargs,
        )
    elif "optimize_positions_and_volume" in tasks:
        results[
            "structure_with_optimized_positions_and_volume"
        ] = optimize_positions_and_volume_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            ase_optimizer=ase_optimizer,
            ase_optimizer_kwargs=ase_optimizer_kwargs,
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            output=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The ASE calculator does not implement:", tasks)
    return results


def calc_static_with_ase(
    structure,
    ase_calculator,
    output=OutputStatic.fields(),
):
    return ASEOutputStatic.get(
        ASEExecutor(ase_structure=structure, ase_calculator=ase_calculator), *output
    )


def calc_molecular_dynamics_langevin_with_ase(
    structure,
    ase_calculator,
    run=100,
    thermo=100,
    timestep=1 * units.fs,
    temperature=100,
    friction=0.002,
    quantities=ASEOutputMolecularDynamics.fields(),
):
    structure.calc = ase_calculator
    MaxwellBoltzmannDistribution(atoms=structure, temperature_K=temperature)
    dyn = Langevin(
        atoms=structure, timestep=timestep, temperature_K=temperature, friction=friction
    )
    loops_to_execute = int(run / thermo)
    cache = {q: [] for q in quantities}
    for i in range(loops_to_execute):
        dyn.run(thermo)
        calc_dict = ASEOutputMolecularDynamics.get(
            ASEExecutor(ase_structure=structure, ase_calculator=ase_calculator),
            *quantities,
        )
        for k, v in calc_dict.items():
            cache[k].append(v)
    return {q: np.array(cache[q]) for q in quantities}


def optimize_positions_with_ase(
    structure, ase_calculator, ase_optimizer, ase_optimizer_kwargs
):
    structure_optimized = structure.copy()
    structure_optimized.calc = ase_calculator
    ase_optimizer_obj = ase_optimizer(structure_optimized)
    ase_optimizer_obj.run(**ase_optimizer_kwargs)
    return structure_optimized


def optimize_positions_and_volume_with_ase(
    structure, ase_calculator, ase_optimizer, ase_optimizer_kwargs
):
    structure_optimized = structure.copy()
    structure_optimized.calc = ase_calculator
    ase_optimizer_obj = ase_optimizer(UnitCellFilter(structure_optimized))
    ase_optimizer_obj.run(**ase_optimizer_kwargs)
    return structure_optimized
