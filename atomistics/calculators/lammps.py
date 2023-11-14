from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Template
from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.lammps_potentials import (
    update_potential_paths,
    view_potentials,
)
from atomistics.calculators.wrapper import as_task_dict_evaluator

if TYPE_CHECKING:
    from ase import Atoms
    from pandas import DataFrame
    from pylammpsmpi import LammpsASELibrary
    from atomistics.calculators.interface import TaskName


LAMMPS_STATIC_RUN_INPUT_TEMPLATE = """\
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo 100
run 0"""


LAMMPS_MINIMIZE_POSITIONS_INPUT_TEMPLATE = """\
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo 10
min_style {{min_style}}
minimize {{etol}} {{ftol}} {{maxiter}} {{maxeval}}"""


LAMMPS_MINIMIZE_POSITIONS_AND_VOLUME_INPUT_TEMPLATE = """\
fix ensemble all box/relax iso 0.0
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo 10
min_style {{min_style}}
minimize {{etol}} {{ftol}} {{maxiter}} {{maxeval}}"""


def template_render(
    template_str,
    min_style="cg",
    etol=0.0,
    ftol=0.0001,
    maxiter=100000,
    maxeval=10000000,
):
    return Template(template_str).render(
        min_style=min_style, etol=etol, ftol=ftol, maxiter=maxiter, maxeval=maxeval
    )


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
        lmp = _run_simulation(
            structure=structure,
            potential_dataframe=potential_dataframe,
            input_template=template_render(
                template_str=LAMMPS_MINIMIZE_POSITIONS_AND_VOLUME_INPUT_TEMPLATE,
                **lmp_optimizer_kwargs,
            ),
            lmp=lmp,
        )
        structure_copy = structure.copy()
        structure_copy.set_cell(lmp.interactive_cells_getter(), scale_atoms=True)
        structure_copy.positions = lmp.interactive_positions_getter()
        results["structure_with_optimized_positions_and_volume"] = structure_copy
    elif "optimize_positions" in tasks:
        lmp = _run_simulation(
            structure=structure,
            potential_dataframe=potential_dataframe,
            input_template=template_render(
                template_str=LAMMPS_MINIMIZE_POSITIONS_INPUT_TEMPLATE,
                **lmp_optimizer_kwargs,
            ),
            lmp=lmp,
        )
        structure_copy = structure.copy()
        structure_copy.positions = lmp.interactive_positions_getter()
        results["structure_with_optimized_positions"] = structure_copy
    elif "calc_energy" in tasks or "calc_forces" in tasks:
        lmp = _run_simulation(
            structure=structure,
            potential_dataframe=potential_dataframe,
            input_template=LAMMPS_STATIC_RUN_INPUT_TEMPLATE,
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


def _run_simulation(structure, potential_dataframe, input_template, lmp):
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


def get_potential_dataframe(structure, resource_path):
    return update_potential_paths(
        df_pot=view_potentials(structure=structure, resource_path=resource_path),
        resource_path=resource_path,
    )
