from __future__ import annotations

from typing import TYPE_CHECKING

from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.wrapper import task_evaluation, TaskName
from atomistics.calculators.lammps_library.calculator import (
    _run_simulation,
    lammps_input_template as _lammps_input_template,
)

if TYPE_CHECKING:
    from ase import Atoms
    from pandas import DataFrame
    from pylammpsmpi import LammpsASELibrary


@task_evaluation
def evaluate_with_lammps_library(
    structure: Atoms,
    tasks: list[TaskName],
    potential_dataframe: DataFrame,
    lmp: LammpsASELibrary,
):
    lmp = _run_simulation(structure, potential_dataframe, _lammps_input_template, lmp)
    results = {}
    if "calc_energy" in tasks:
        results["energy"] = lmp.interactive_energy_pot_getter()
    if "calc_forces" in tasks:
        results["forces"] = lmp.interactive_forces_getter()
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
    results_dict = evaluate_with_lammps_library(task_dict, potential_dataframe, lmp)
    lmp.close()
    return results_dict
