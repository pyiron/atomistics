from __future__ import annotations

from jinja2 import Template
import numpy as np
from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.lammps.potential import validate_potential_dataframe
from atomistics.calculators.lammps.output import LammpsOutputMolecularDynamics
from atomistics.shared.thermal_expansion import (
    OutputThermalExpansionProperties,
    ThermalExpansionProperties,
)
from atomistics.shared.tqdm_iterator import get_tqdm_iterator


def lammps_run(structure, potential_dataframe, input_template=None, lmp=None, **kwargs):
    potential_dataframe = validate_potential_dataframe(
        potential_dataframe=potential_dataframe
    )
    if lmp is None:
        lmp = LammpsASELibrary(**kwargs)

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

    if input_template is not None:
        for l in input_template.split("\n"):
            lmp.interactive_lib_command(l)

    return lmp


def lammps_calc_md_step(
    lmp_instance,
    run_str,
    run,
    output=LammpsOutputMolecularDynamics.fields(),
):
    run_str_rendered = Template(run_str).render(run=run)
    lmp_instance.interactive_lib_command(run_str_rendered)
    return LammpsOutputMolecularDynamics.get(lmp_instance, *output)


def lammps_calc_md(
    lmp_instance,
    run_str,
    run,
    thermo,
    output=LammpsOutputMolecularDynamics.fields(),
):
    results_lst = [
        lammps_calc_md_step(
            lmp_instance=lmp_instance,
            run_str=run_str,
            run=thermo,
            output=output,
        )
        for _ in range(run // thermo)
    ]
    return {q: np.array([d[q] for d in results_lst]) for q in output}


def lammps_thermal_expansion_loop(
    structure,
    potential_dataframe,
    init_str,
    run_str,
    temperature_lst,
    run=100,
    thermo=100,
    timestep=0.001,
    Tdamp=0.1,
    Pstart=0.0,
    Pstop=0.0,
    Pdamp=1.0,
    seed=4928459,
    dist="gaussian",
    lmp=None,
    output=OutputThermalExpansionProperties.fields(),
    **kwargs,
):
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(init_str).render(
            thermo=thermo,
            temp=temperature_lst[0],
            timestep=timestep,
            seed=seed,
            dist=dist,
        ),
        lmp=lmp,
        **kwargs,
    )

    volume_md_lst, temperature_md_lst = [], []
    for temp in get_tqdm_iterator(temperature_lst):
        run_str_rendered = Template(run_str).render(
            run=run,
            Tstart=temp - 5,
            Tstop=temp,
            Tdamp=Tdamp,
            Pstart=Pstart,
            Pstop=Pstop,
            Pdamp=Pdamp,
        )
        for l in run_str_rendered.split("\n"):
            lmp_instance.interactive_lib_command(l)
        volume_md_lst.append(lmp_instance.interactive_volume_getter())
        temperature_md_lst.append(lmp_instance.interactive_temperatures_getter())
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return OutputThermalExpansionProperties.get(
        ThermalExpansionProperties(
            temperatures_lst=temperature_md_lst, volumes_lst=volume_md_lst
        ),
        *output,
    )


def lammps_shutdown(lmp_instance, close_instance=True):
    lmp_instance.interactive_lib_command("clear")
    if close_instance:
        lmp_instance.close()
