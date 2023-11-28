from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Template
import numpy as np
import pandas
from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.wrapper import as_task_dict_evaluator
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


def template_render_minimize(
    template_str,
    min_style="cg",
    etol=0.0,
    ftol=0.0001,
    maxiter=100000,
    maxeval=10000000,
    thermo=10,
):
    return Template(template_str).render(
        min_style=min_style,
        etol=etol,
        ftol=ftol,
        maxiter=maxiter,
        maxeval=maxeval,
        thermo=thermo,
    )


def template_render_run(
    template_str,
    run=0,
    thermo=100,
):
    return Template(template_str).render(
        run=run,
        thermo=thermo,
    )


def calc_thermal_expansion_md(
    structure, potential_dataframe, lmp, Tstart=15, Tstop=1500, Tstep=5
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

    lmp = _run_simulation(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(init_str).render(
            thermo=100, temp=Tstart, timestep=0.001, seed=4928459, dist="gaussian"
        ),
        lmp=lmp,
    )

    volume_md_lst = []
    temperature_lst = np.arange(Tstart, Tstop + Tstep, Tstep).tolist()
    for temp in temperature_lst:
        run_str_rendered = Template(run_str).render(
            run=100,
            Tstart=temp - 5,
            Tstop=temp,
            Tdamp=0.1,
            Pstart=0.0,
            Pstop=0.0,
            Pdamp=1.0,
        )
        for l in run_str_rendered.split("\n"):
            lmp.interactive_lib_command(l)
        volume_md_lst.append(lmp.interactive_volume_getter())
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
        template_str = (
            LAMMPS_MINIMIZE_VOLUME
            + "\n"
            + LAMMPS_THERMO_STYLE
            + "\n"
            + LAMMPS_THERMO
            + "\n"
            + LAMMPS_MINIMIZE
        )
        lmp = _run_simulation(
            structure=structure,
            potential_dataframe=potential_dataframe,
            input_template=template_render_minimize(
                template_str=template_str,
                **lmp_optimizer_kwargs,
            ),
            lmp=lmp,
        )
        structure_copy = structure.copy()
        structure_copy.set_cell(lmp.interactive_cells_getter(), scale_atoms=True)
        structure_copy.positions = lmp.interactive_positions_getter()
        results["structure_with_optimized_positions_and_volume"] = structure_copy
    elif "optimize_positions" in tasks:
        template_str = (
            LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_MINIMIZE
        )
        lmp = _run_simulation(
            structure=structure,
            potential_dataframe=potential_dataframe,
            input_template=template_render_minimize(
                template_str=template_str,
                **lmp_optimizer_kwargs,
            ),
            lmp=lmp,
        )
        structure_copy = structure.copy()
        structure_copy.positions = lmp.interactive_positions_getter()
        results["structure_with_optimized_positions"] = structure_copy
    elif "calc_molecular_dynamics_thermal_expansion" in tasks:
        temperature_lst, volume_md_lst = calc_thermal_expansion_md(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            **lmp_optimizer_kwargs,
        )
        results["volume_over_temperature"] = (temperature_lst, volume_md_lst)
    elif "calc_energy" in tasks or "calc_forces" in tasks:
        template_str = LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_RUN
        lmp = _run_simulation(
            structure=structure,
            potential_dataframe=potential_dataframe,
            input_template=template_render_run(
                template_str=template_str,
                thermo=100,
                run=0,
            ),
            lmp=lmp,
        )
        if "calc_energy" in tasks:
            results["energy"] = lmp.interactive_energy_pot_getter()
        if "calc_forces" in tasks:
            results["forces"] = lmp.interactive_forces_getter()
    else:
        raise ValueError("The LAMMPS calculator does not implement:", tasks)
    lmp.interactive_lib_command("clear")
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


def validate_potential_dataframe(potential_dataframe):
    if isinstance(potential_dataframe, pandas.Series):
        return potential_dataframe
    elif isinstance(potential_dataframe, pandas.DataFrame):
        if len(potential_dataframe) == 1:
            return potential_dataframe.iloc[0]
        elif len(potential_dataframe) == 0:
            raise ValueError(
                "The potential_dataframe is an empty pandas.DataFrame:",
                potential_dataframe,
            )
        else:
            raise ValueError(
                "The potential_dataframe contains more than one interatomic potential, please select one:",
                potential_dataframe,
            )
    else:
        raise TypeError(
            "The potential_dataframe should be a pandas.DataFrame or pandas.Series, but instead it is of type:",
            type(potential_dataframe),
        )


def _run_simulation(structure, potential_dataframe, input_template, lmp):
    potential_dataframe = validate_potential_dataframe(
        potential_dataframe=potential_dataframe
    )

    # write structure to LAMMPS
    lmp.interactive_structure_setter(
        structure=structure,
        units="metal",
        dimension=3,
        boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
        atom_style="atomic",
        el_eam_lst=potential_dataframe.Species,
        calc_md=False,
    )

    # execute calculation
    for c in potential_dataframe.Config:
        lmp.interactive_lib_command(c)

    for l in input_template.split("\n"):
        lmp.interactive_lib_command(l)

    return lmp
