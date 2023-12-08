from __future__ import annotations

from jinja2 import Template
import numpy as np
from typing import TYPE_CHECKING

from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.calculators.lammps.helpers import (
    lammps_calc_md,
    lammps_run,
    lammps_thermal_expansion_loop,
    lammps_shutdown,
)
from atomistics.calculators.lammps.commands import (
    LAMMPS_THERMO_STYLE,
    LAMMPS_THERMO,
    LAMMPS_ENSEMBLE_NPT,
    LAMMPS_ENSEMBLE_NPH,
    LAMMPS_ENSEMBLE_NVT,
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
        input_template=Template(template_str).render(
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
        input_template=Template(template_str).render(
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


def calc_static_with_lammps(
    structure,
    potential_dataframe,
    lmp=None,
    quantities=("energy", "forces", "stress"),
    **kwargs,
):
    template_str = LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_RUN
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(template_str).render(
            run=0,
            thermo=100,
        ),
        lmp=lmp,
        **kwargs,
    )
    interactive_getter_dict = {
        "forces": lmp_instance.interactive_forces_getter,
        "energy": lmp_instance.interactive_energy_pot_getter,
        "stress": lmp_instance.interactive_pressures_getter,
    }
    result_dict = {q: interactive_getter_dict[q]() for q in quantities}
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_nvt_with_lammps(
    structure,
    potential_dataframe,
    Tstart=100,
    Tstop=100,
    Tdamp=0.1,
    run=100,
    thermo=10,
    timestep=0.001,
    seed=4928459,
    dist="gaussian",
    lmp=None,
    quantities=(
        "positions",
        "cell",
        "forces",
        "temperature",
        "energy_pot",
        "energy_tot",
        "pressure",
        "velocities",
    ),
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
        + LAMMPS_ENSEMBLE_NVT
    )
    run_str = LAMMPS_RUN + "\n"
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(init_str).render(
            thermo=thermo,
            Tstart=Tstart,
            temp=Tstart,
            Tstop=Tstop,
            Tdamp=Tdamp,
            timestep=timestep,
            seed=seed,
            dist=dist,
        ),
        lmp=lmp,
        **kwargs,
    )
    result_dict = lammps_calc_md(
        lmp_instance=lmp_instance,
        run_str=run_str,
        run=run,
        thermo=thermo,
        quantities=quantities,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_npt_with_lammps(
    structure,
    potential_dataframe,
    Tstart=100,
    Tstop=100,
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
    quantities=(
        "positions",
        "cell",
        "forces",
        "temperature",
        "energy_pot",
        "energy_tot",
        "pressure",
        "velocities",
    ),
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
        + LAMMPS_ENSEMBLE_NPT
    )
    run_str = LAMMPS_RUN + "\n"
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(init_str).render(
            thermo=thermo,
            Tstart=Tstart,
            temp=Tstart,
            Tstop=Tstop,
            Tdamp=Tdamp,
            Pstart=Pstart,
            Pstop=Pstop,
            Pdamp=Pdamp,
            timestep=timestep,
            seed=seed,
            dist=dist,
        ),
        lmp=lmp,
        **kwargs,
    )
    result_dict = lammps_calc_md(
        lmp_instance=lmp_instance,
        run_str=run_str,
        run=run,
        thermo=thermo,
        quantities=quantities,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_nph_with_lammps(
    structure,
    potential_dataframe,
    run=100,
    thermo=100,
    timestep=0.001,
    Tstart=100,
    Pstart=0.0,
    Pstop=0.0,
    Pdamp=1.0,
    seed=4928459,
    dist="gaussian",
    lmp=None,
    quantities=(
        "positions",
        "cell",
        "forces",
        "temperature",
        "energy_pot",
        "energy_tot",
        "pressure",
        "velocities",
    ),
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
        + LAMMPS_ENSEMBLE_NPH
    )
    run_str = LAMMPS_RUN + "\n"
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(init_str).render(
            thermo=thermo,
            temp=Tstart,
            Pstart=Pstart,
            Pstop=Pstop,
            Pdamp=Pdamp,
            timestep=timestep,
            seed=seed,
            dist=dist,
        ),
        lmp=lmp,
        **kwargs,
    )
    result_dict = lammps_calc_md(
        lmp_instance=lmp_instance,
        run_str=run_str,
        run=run,
        thermo=thermo,
        quantities=quantities,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


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
    temperature_md_lst, volume_md_lst = lammps_thermal_expansion_loop(
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
    return temperature_md_lst, volume_md_lst


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
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        quantities = []
        if "calc_energy" in tasks:
            quantities.append("energy")
        if "calc_forces" in tasks:
            quantities.append("forces")
        if "calc_stress" in tasks:
            quantities.append("stress")
        return calc_static_with_lammps(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            quantities=quantities,
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
