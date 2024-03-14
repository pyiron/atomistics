from __future__ import annotations

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import UnitCellFilter
from ase.calculators.calculator import PropertyNotImplementedError
import numpy as np
from typing import TYPE_CHECKING

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import (
    OutputStatic,
    OutputMolecularDynamics,
    OutputThermalExpansion,
)
from atomistics.shared.thermal_expansion import get_thermal_expansion_output
from atomistics.shared.tqdm_iterator import get_tqdm_iterator

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from ase.calculators.calculator import Calculator as ASECalculator
    from ase.optimize.optimize import Optimizer
    from atomistics.calculators.interface import TaskName


class ASEExecutor(object):
    def __init__(self, ase_structure, ase_calculator):
        self.structure = ase_structure
        self.structure.calc = ase_calculator

    def forces(self) -> np.ndarray:
        return self.structure.get_forces()

    def energy(self) -> float:
        return self.structure.get_potential_energy()

    def energy_pot(self) -> float:
        return self.structure.get_potential_energy()

    def energy_tot(self) -> float:
        return (
            self.structure.get_potential_energy() + self.structure.get_kinetic_energy()
        )

    def stress(self) -> np.ndarray:
        try:
            return self.structure.get_stress(voigt=False)
        except PropertyNotImplementedError:
            return None

    def pressure(self) -> np.ndarray:
        try:
            return self.structure.get_stress(voigt=False)
        except PropertyNotImplementedError:
            return None

    def cell(self) -> np.ndarray:
        return self.structure.get_cell()

    def positions(self) -> np.ndarray:
        return self.structure.get_positions()

    def velocities(self) -> np.ndarray:
        return self.structure.get_velocities()

    def temperature(self) -> float:
        return self.structure.get_temperature()

    def volume(self) -> float:
        return self.structure.get_volume()


@as_task_dict_evaluator
def evaluate_with_ase(
    structure: Atoms,
    tasks: list[TaskName],
    ase_calculator: ASECalculator,
    ase_optimizer: Optimizer = None,
    ase_optimizer_kwargs: dict = {},
) -> dict:
    results = {}
    if "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = optimize_positions_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            ase_optimizer=ase_optimizer,
            ase_optimizer_kwargs=ase_optimizer_kwargs,
        )
    elif "optimize_positions_and_volume" in tasks:
        results["structure_with_optimized_positions_and_volume"] = (
            optimize_positions_and_volume_with_ase(
                structure=structure,
                ase_calculator=ase_calculator,
                ase_optimizer=ase_optimizer,
                ase_optimizer_kwargs=ase_optimizer_kwargs,
            )
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The ASE calculator does not implement:", tasks)
    return results


def calc_static_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    output_keys=OutputStatic.keys(),
):
    ase_exe = ASEExecutor(ase_structure=structure, ase_calculator=ase_calculator)
    return OutputStatic(**{k: getattr(ase_exe, k) for k in OutputStatic.keys()}).get(
        output_keys=output_keys
    )


def calc_molecular_dynamics_npt_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 1 * units.fs,
    ttime: float = 100 * units.fs,
    pfactor: float = 2e6 * units.GPa * (units.fs**2),
    temperature: float = 100.0,
    externalstress: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar,
    output_keys=OutputMolecularDynamics.keys(),
) -> dict:
    return _calc_molecular_dynamics_with_ase(
        dyn=NPT(
            atoms=structure,
            timestep=timestep,
            temperature=None,
            externalstress=externalstress,
            ttime=ttime,
            pfactor=pfactor,
            temperature_K=temperature,
            mask=None,
            trajectory=None,
            logfile=None,
            loginterval=1,
            append_trajectory=False,
        ),
        structure=structure,
        ase_calculator=ase_calculator,
        temperature=temperature,
        run=run,
        thermo=thermo,
        output_keys=output_keys,
    )


def calc_molecular_dynamics_langevin_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 1 * units.fs,
    temperature: float = 100.0,
    friction: float = 0.002,
    output_keys=OutputMolecularDynamics.keys(),
):
    return _calc_molecular_dynamics_with_ase(
        dyn=Langevin(
            atoms=structure,
            timestep=timestep,
            temperature_K=temperature,
            friction=friction,
        ),
        structure=structure,
        ase_calculator=ase_calculator,
        temperature=temperature,
        run=run,
        thermo=thermo,
        output_keys=output_keys,
    )


def optimize_positions_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    ase_optimizer: Optimizer,
    ase_optimizer_kwargs: dict,
):
    structure_optimized = structure.copy()
    structure_optimized.calc = ase_calculator
    ase_optimizer_obj = ase_optimizer(structure_optimized)
    ase_optimizer_obj.run(**ase_optimizer_kwargs)
    return structure_optimized


def optimize_positions_and_volume_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    ase_optimizer: Optimizer,
    ase_optimizer_kwargs: dict,
):
    structure_optimized = structure.copy()
    structure_optimized.calc = ase_calculator
    ase_optimizer_obj = ase_optimizer(UnitCellFilter(structure_optimized))
    ase_optimizer_obj.run(**ase_optimizer_kwargs)
    return structure_optimized


def calc_molecular_dynamics_thermal_expansion_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    temperature_start: float = 15.0,
    temperature_stop: float = 1500.0,
    temperature_step: float = 5.0,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 1 * units.fs,
    ttime: float = 100 * units.fs,
    pfactor: float = 2e6 * units.GPa * (units.fs**2),
    externalstress: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar,
    output_keys=OutputThermalExpansion.keys(),
):
    structure_current = structure.copy()
    temperature_lst = np.arange(
        temperature_start, temperature_stop + temperature_step, temperature_step
    ).tolist()
    volume_md_lst, temperature_md_lst = [], []
    for temperature in get_tqdm_iterator(temperature_lst):
        result_dict = calc_molecular_dynamics_npt_with_ase(
            structure=structure_current.copy(),
            ase_calculator=ase_calculator,
            run=run,
            thermo=thermo,
            timestep=timestep,
            ttime=ttime,
            pfactor=pfactor,
            temperature=temperature,
            externalstress=externalstress,
        )
        structure_current.set_cell(cell=result_dict["cell"][-1], scale_atoms=True)
        temperature_md_lst.append(result_dict["temperature"][-1])
        volume_md_lst.append(result_dict["volume"][-1])
    return get_thermal_expansion_output(
        temperatures_lst=temperature_md_lst,
        volumes_lst=volume_md_lst,
        output_keys=output_keys,
    )


def _calc_molecular_dynamics_with_ase(
    dyn,
    structure: Atoms,
    ase_calculator: ASECalculator,
    temperature: float,
    run: int,
    thermo: int,
    output_keys: tuple[str],
):
    structure.calc = ase_calculator
    MaxwellBoltzmannDistribution(atoms=structure, temperature_K=temperature)
    cache = {q: [] for q in output_keys}
    for i in range(int(run / thermo)):
        dyn.run(thermo)
        ase_instance = ASEExecutor(
            ase_structure=structure, ase_calculator=ase_calculator
        )
        calc_dict = OutputMolecularDynamics(
            **{k: getattr(ase_instance, k) for k in OutputMolecularDynamics.keys()}
        ).get(output_keys=output_keys)
        for k, v in calc_dict.items():
            cache[k].append(v)
    return {q: np.array(cache[q]) for q in output_keys}
