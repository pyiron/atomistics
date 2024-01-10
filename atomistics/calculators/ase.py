from __future__ import annotations

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import UnitCellFilter
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
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The ASE calculator does not implement:", tasks)
    return results


def calc_static_with_ase(
    structure,
    ase_calculator,
    output_keys=OutputStatic.keys(),
):
    structure.calc = ase_calculator
    output_dict = {}
    if "forces" in output_keys:
        output_dict["forces"] = structure.get_forces()
    if "energy" in output_keys:
        output_dict["energy"] = structure.get_potential_energy()
    if "stress" in output_keys:
        output_dict["stress"] = structure.get_stress(voigt=False)
    if "volume" in output_keys:
        output_dict["volume"] = structure.get_volume()
    return output_dict


def calc_molecular_dynamics_npt_with_ase(
    structure,
    ase_calculator,
    run=100,
    thermo=100,
    timestep=1 * units.fs,
    ttime=100 * units.fs,
    pfactor=2e6 * units.GPa * (units.fs**2),
    temperature=100,
    externalstress=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar,
    output_keys=OutputMolecularDynamics.keys(),
):
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
    structure,
    ase_calculator,
    run=100,
    thermo=100,
    timestep=1 * units.fs,
    temperature=100,
    friction=0.002,
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


def calc_molecular_dynamics_thermal_expansion_with_ase(
    structure,
    ase_calculator,
    temperature_start=15,
    temperature_stop=1500,
    temperature_step=5,
    run=100,
    thermo=100,
    timestep=1 * units.fs,
    ttime=100 * units.fs,
    pfactor=2e6 * units.GPa * (units.fs**2),
    externalstress=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar,
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
    dyn, structure, ase_calculator, temperature, run, thermo, output_keys
):
    structure.calc = ase_calculator
    MaxwellBoltzmannDistribution(atoms=structure, temperature_K=temperature)
    cache = {q: [] for q in output_keys}
    for i in range(int(run / thermo)):
        dyn.run(thermo)
        if "positions" in output_keys:
            cache["positions"].append(structure.get_positions())
        if "cell" in output_keys:
            cache["cell"].append(structure.get_cell())
        if "forces" in output_keys:
            cache["forces"].append(structure.get_forces())
        if "temperature" in output_keys:
            cache["temperature"].append(structure.get_temperature())
        if "energy_pot" in output_keys:
            cache["energy_pot"].append(structure.get_potential_energy())
        if "energy_tot" in output_keys:
            cache["energy_tot"].append(
                structure.get_potential_energy() + structure.get_kinetic_energy()
            )
        if "pressure" in output_keys:
            cache["pressure"].append(structure.get_stress(voigt=False))
        if "velocities" in output_keys:
            cache["velocities"].append(structure.get_velocities())
        if "volume" in output_keys:
            cache["volume"].append(structure.get_volume())
    return {q: np.array(cache[q]) for q in output_keys}
