from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.calculators.lammps.helpers import (
    lammps_run,
    lammps_thermal_expansion_loop,
    lammps_shutdown,
    template_render_minimize,
    template_render_run,
)
from atomistics.calculators.lammps.commands import (
    LAMMPS_THERMO_STYLE,
    LAMMPS_THERMO,
    LAMMPS_ENSEMBLE_NPT,
    LAMMPS_VELOCITY,
    LAMMPS_TIMESTEP,
    LAMMPS_MINIMIZE,
    LAMMPS_RUN,
    LAMMPS_MINIMIZE_VOLUME,
)

if TYPE_CHECKING:
    from ase import Atoms
    from pandas import DataFrame
    from pylammpsmpi import LammpsASELibrary
    from atomistics.calculators.interface import TaskName


def calc_energy_with_lammps(
    structure: Atoms, potential_dataframe: DataFrame, lmp=None, **kwargs
):
    template_str = LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_RUN
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=template_render_run(
            template_str=template_str,
            thermo=100,
            run=0,
        ),
        lmp=lmp,
        **kwargs,
    )
    energy_pot = lmp_instance.interactive_energy_pot_getter()
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return energy_pot


def calc_forces_with_lammps(
    structure: Atoms, potential_dataframe: DataFrame, lmp=None, **kwargs
):
    template_str = LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_RUN
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=template_render_run(
            template_str=template_str,
            thermo=100,
            run=0,
        ),
        lmp=lmp,
        **kwargs,
    )
    forces = lmp_instance.interactive_forces_getter()
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return forces


def calc_energy_and_forces_with_lammps(
    structure, potential_dataframe: DataFrame, lmp=None, **kwargs
):
    template_str = LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_RUN
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=template_render_run(
            template_str=template_str,
            thermo=100,
            run=0,
        ),
        lmp=lmp,
        **kwargs,
    )
    energy_pot = lmp_instance.interactive_energy_pot_getter()
    forces = lmp_instance.interactive_forces_getter()
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return energy_pot, forces


def optimize_positions_and_volume_with_lammps(
    structure: Atoms,
    potential_dataframe: DataFrame,
    min_style="cg",
    etol=0.0,
    ftol=0.0001,
    maxiter=100000,
    maxeval=10000000,
    thermo=10,
    lmp=None,
    **kwargs,
):
    template_str = (
        LAMMPS_MINIMIZE_VOLUME
        + "\n"
        + LAMMPS_THERMO_STYLE
        + "\n"
        + LAMMPS_THERMO
        + "\n"
        + LAMMPS_MINIMIZE
    )
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=template_render_minimize(
            template_str=template_str,
            min_style=min_style,
            etol=etol,
            ftol=ftol,
            maxiter=maxiter,
            maxeval=maxeval,
            thermo=thermo,
        ),
        lmp=lmp,
        **kwargs,
    )
    structure_copy = structure.copy()
    structure_copy.set_cell(lmp_instance.interactive_cells_getter(), scale_atoms=True)
    structure_copy.positions = lmp_instance.interactive_positions_getter()
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return structure_copy


def optimize_positions_with_lammps(
    structure: Atoms,
    potential_dataframe: DataFrame,
    min_style="cg",
    etol=0.0,
    ftol=0.0001,
    maxiter=100000,
    maxeval=10000000,
    thermo=10,
    lmp=None,
    **kwargs,
):
    template_str = LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_MINIMIZE
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=template_render_minimize(
            template_str=template_str,
            min_style=min_style,
            etol=etol,
            ftol=ftol,
            maxiter=maxiter,
            maxeval=maxeval,
            thermo=thermo,
        ),
        lmp=lmp,
        **kwargs,
    )
    structure_copy = structure.copy()
    structure_copy.positions = lmp_instance.interactive_positions_getter()
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return structure_copy


def calc_molecular_dynamics_thermal_expansion_with_lammps(
    structure,
    potential_dataframe,
    Tstart=15,
    Tstop=1500,
    Tstep=5,
    Tdamp=0.1,
    run=100,
    thermo=100,
    timestep=0.001,
    Pstart=0.0,
    Pstop=0.0,
    Pdamp=1.0,
    seed=4928459,
    dist="gaussian",
    lmp=None,
    **kwargs,
):
    init_str = (
        LAMMPS_THERMO_STYLE
        + "\n"
        + LAMMPS_TIMESTEP
        + "\n"
        + LAMMPS_THERMO
        + "\n"
        + LAMMPS_VELOCITY
        + "\n"
    )
    run_str = LAMMPS_ENSEMBLE_NPT + "\n" + LAMMPS_RUN
    temperature_lst = np.arange(Tstart, Tstop + Tstep, Tstep).tolist()
    volume_md_lst = lammps_thermal_expansion_loop(
        structure=structure,
        potential_dataframe=potential_dataframe,
        init_str=init_str,
        run_str=run_str,
        temperature_lst=temperature_lst,
        run=run,
        thermo=thermo,
        timestep=timestep,
        Tdamp=Tdamp,
        Pstart=Pstart,
        Pstop=Pstop,
        Pdamp=Pdamp,
        seed=seed,
        dist=dist,
        lmp=lmp,
        **kwargs,
    )
    return temperature_lst, volume_md_lst


@as_task_dict_evaluator
def evaluate_with_lammps_library(
    structure: Atoms,
    tasks: list[TaskName],
    potential_dataframe: DataFrame,
    lmp: LammpsASELibrary,
    lmp_optimizer_kwargs: dict = {},
):
    results = {}
    if "optimize_positions_and_volume" in tasks:
        results[
            "structure_with_optimized_positions_and_volume"
        ] = optimize_positions_and_volume_with_lammps(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            **lmp_optimizer_kwargs,
        )
    elif "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = optimize_positions_with_lammps(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            **lmp_optimizer_kwargs,
        )
    elif "calc_molecular_dynamics_thermal_expansion" in tasks:
        (
            temperature_lst,
            volume_md_lst,
        ) = calc_molecular_dynamics_thermal_expansion_with_lammps(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            **lmp_optimizer_kwargs,
        )
        results["volume_over_temperature"] = (temperature_lst, volume_md_lst)
    elif "calc_energy" in tasks and "calc_forces" in tasks:
        results["energy"], results["forces"] = calc_energy_and_forces_with_lammps(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
        )
    elif "calc_energy" in tasks:
        results["energy"] = calc_energy_with_lammps(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
        )
    elif "calc_forces" in tasks:
        results["forces"] = calc_forces_with_lammps(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
        )
    else:
        raise ValueError("The LAMMPS calculator does not implement:", tasks)
    return results


def evaluate_with_lammps(
    task_dict: dict[str, dict[str, Atoms]],
    potential_dataframe: DataFrame,
    working_directory=None,
    cores=1,
    comm=None,
    logger=None,
    log_file=None,
    library=None,
    diable_log_file=True,
    lmp_optimizer_kwargs={},
):
    lmp = LammpsASELibrary(
        working_directory=working_directory,
        cores=cores,
        comm=comm,
        logger=logger,
        log_file=log_file,
        library=library,
        diable_log_file=diable_log_file,
    )
    results_dict = evaluate_with_lammps_library(
        task_dict=task_dict,
        potential_dataframe=potential_dataframe,
        lmp=lmp,
        lmp_optimizer_kwargs=lmp_optimizer_kwargs,
    )
    lmp.close()
    return results_dict
