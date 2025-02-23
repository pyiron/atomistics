from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas
from jinja2 import Template
from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.lammps.commands import (
    LAMMPS_ENSEMBLE_NPH,
    LAMMPS_ENSEMBLE_NPT,
    LAMMPS_ENSEMBLE_NVT,
    LAMMPS_LANGEVIN,
    LAMMPS_MINIMIZE,
    LAMMPS_MINIMIZE_VOLUME,
    LAMMPS_NVE,
    LAMMPS_RUN,
    LAMMPS_THERMO,
    LAMMPS_THERMO_STYLE,
    LAMMPS_TIMESTEP,
    LAMMPS_VELOCITY,
)
from atomistics.calculators.lammps.helpers import (
    lammps_calc_md,
    lammps_run,
    lammps_shutdown,
    lammps_thermal_expansion_loop,
)
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputMolecularDynamics, OutputStatic
from atomistics.shared.thermal_expansion import OutputThermalExpansion

if TYPE_CHECKING:
    from ase import Atoms
    from pandas import DataFrame
    from pylammpsmpi import LammpsASELibrary

    from atomistics.calculators.interface import TaskName


def optimize_positions_and_volume_with_lammpslib(
    structure: Atoms,
    potential_dataframe: DataFrame,
    min_style: str = "cg",
    etol: float = 0.0,
    ftol: float = 0.0001,
    maxiter: int = 100000,
    maxeval: int = 10000000,
    thermo: int = 10,
    lmp=None,
    **kwargs,
) -> Atoms:
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


def optimize_positions_with_lammpslib(
    structure: Atoms,
    potential_dataframe: DataFrame,
    min_style: str = "cg",
    etol: float = 0.0,
    ftol: float = 0.0001,
    maxiter: int = 100000,
    maxeval: int = 10000000,
    thermo: int = 10,
    lmp=None,
    **kwargs,
) -> Atoms:
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


def calc_static_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    lmp=None,
    output_keys=OutputStatic.keys(),
    **kwargs,
) -> dict:
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
    result_dict = OutputStatic(
        forces=lmp_instance.interactive_forces_getter,
        energy=lmp_instance.interactive_energy_pot_getter,
        stress=lmp_instance.interactive_pressures_getter,
        volume=lmp_instance.interactive_volume_getter,
    ).get(output_keys=output_keys)
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_nvt_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    Tstart: float = 100.0,
    Tstop: float = 100.0,
    Tdamp: float = 0.1,
    run: int = 100,
    thermo: int = 10,
    timestep: float = 0.001,
    seed: int = 4928459,
    dist: str = "gaussian",
    lmp=None,
    output_keys=OutputMolecularDynamics.keys(),
    **kwargs,
) -> dict:
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
        output_keys=output_keys,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_npt_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    Tstart: float = 100.0,
    Tstop: float = 100.0,
    Tdamp: float = 0.1,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Pstart: float = 0.0,
    Pstop: float = 0.0,
    Pdamp: float = 1.0,
    seed: int = 4928459,
    dist: str = "gaussian",
    lmp=None,
    output_keys=OutputMolecularDynamics.keys(),
    **kwargs,
) -> dict:
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
        output_keys=output_keys,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_nph_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Tstart: float = 100.0,
    Pstart: float = 0.0,
    Pstop: float = 0.0,
    Pdamp: float = 1.0,
    seed: int = 4928459,
    dist: str = "gaussian",
    lmp=None,
    output_keys=OutputMolecularDynamics.keys(),
    **kwargs,
) -> dict:
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
        output_keys=output_keys,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_langevin_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Tstart: float = 100.0,
    Tstop: float = 100,
    Tdamp: float = 0.1,
    seed: int = 4928459,
    dist: str = "gaussian",
    lmp=None,
    output_keys=OutputMolecularDynamics.keys(),
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
        + LAMMPS_NVE
        + "\n"
        + LAMMPS_LANGEVIN
    )
    run_str = LAMMPS_RUN + "\n"
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(init_str).render(
            thermo=thermo,
            temp=Tstart,
            Tstart=Tstart,
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
        output_keys=output_keys,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_thermal_expansion_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    Tstart: float = 15.0,
    Tstop: float = 1500.0,
    Tstep: int = 5,
    Tdamp: float = 0.1,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Pstart: float = 0.0,
    Pstop: float = 0.0,
    Pdamp: float = 1.0,
    seed: int = 4928459,
    dist: str = "gaussian",
    lmp=None,
    output_keys=OutputThermalExpansion.keys(),
    **kwargs,
) -> dict:
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
    return lammps_thermal_expansion_loop(
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
        output_keys=output_keys,
        **kwargs,
    )


@as_task_dict_evaluator
def evaluate_with_lammpslib_library_interface(
    structure: Atoms,
    tasks: list[TaskName],
    potential_dataframe: DataFrame,
    lmp: LammpsASELibrary,
    lmp_optimizer_kwargs: dict = None,
) -> dict:
    if lmp_optimizer_kwargs is None:
        lmp_optimizer_kwargs = {}
    results = {}
    if "optimize_positions_and_volume" in tasks:
        results["structure_with_optimized_positions_and_volume"] = (
            optimize_positions_and_volume_with_lammpslib(
                structure=structure,
                potential_dataframe=potential_dataframe,
                lmp=lmp,
                **lmp_optimizer_kwargs,
            )
        )
    elif "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = (
            optimize_positions_with_lammpslib(
                structure=structure,
                potential_dataframe=potential_dataframe,
                lmp=lmp,
                **lmp_optimizer_kwargs,
            )
        )
    elif "calc_molecular_dynamics_thermal_expansion" in tasks:
        results_dict = calc_molecular_dynamics_thermal_expansion_with_lammpslib(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            **lmp_optimizer_kwargs,
        )
        results["volume_over_temperature"] = (
            results_dict["temperatures"],
            results_dict["volumes"],
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_lammpslib(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The LAMMPS calculator does not implement:", tasks)
    return results


def evaluate_with_lammpslib(
    task_dict: dict[str, dict[str, Atoms]],
    potential_dataframe: DataFrame,
    working_directory=None,
    cores: int = 1,
    comm=None,
    logger=None,
    log_file=None,
    library=None,
    disable_log_file: bool = True,
    lmp_optimizer_kwargs=None,
) -> dict:
    if lmp_optimizer_kwargs is None:
        lmp_optimizer_kwargs = {}
    lmp = LammpsASELibrary(
        working_directory=working_directory,
        cores=cores,
        comm=comm,
        logger=logger,
        log_file=log_file,
        library=library,
        disable_log_file=disable_log_file,
    )
    results_dict = evaluate_with_lammpslib_library_interface(
        task_dict=task_dict,
        potential_dataframe=potential_dataframe,
        lmp=lmp,
        lmp_optimizer_kwargs=lmp_optimizer_kwargs,
    )
    lmp.close()
    return results_dict
