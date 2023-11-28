from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Template
import pandas
from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.calculators.lammps.commands import (
    LAMMPS_THERMO_STYLE,
    LAMMPS_THERMO,
    LAMMPS_MINIMIZE,
    LAMMPS_RUN,
    LAMMPS_MINIMIZE_VOLUME,
)
from atomistics.calculators.lammps.potential import validate_potential_dataframe


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


def lammps_run(structure, potential_dataframe, input_template, lmp=None):
    potential_dataframe = validate_potential_dataframe(
        potential_dataframe=potential_dataframe
    )
    if lmp is None:
        lmp = LammpsASELibrary()

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


def lammps_shutdown(lmp_instance, close_instance=True):
    lmp_instance.interactive_lib_command("clear")
    if close_instance:
        lmp_instance.close()
