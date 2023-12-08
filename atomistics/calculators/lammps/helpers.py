from __future__ import annotations

from jinja2 import Template
import numpy as np
from pylammpsmpi import LammpsASELibrary

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


def lammps_run(structure, potential_dataframe, input_template, lmp=None, **kwargs):
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

    for l in input_template.split("\n"):
        lmp.interactive_lib_command(l)

    return lmp


def lammps_md_nvt(
    structure,
    potential_dataframe,
    init_str,
    run_str,
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
    ),
    **kwargs,
):
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


def lammps_md_npt(
    structure,
    potential_dataframe,
    init_str,
    run_str,
    Tstart=100,
    Tstop=100,
    Tdamp=0.1,
    Pstart=0.0,
    Pstop=0.0,
    Pdamp=1.0,
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
    ),
    **kwargs,
):
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


def lammps_md_nph(
    structure,
    potential_dataframe,
    init_str,
    run_str,
    Tstart=100,
    Pstart=0.0,
    Pstop=0.0,
    Pdamp=1.0,
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
    ),
    **kwargs,
):
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


def lammps_calc_md_step(
    lmp_instance,
    run_str,
    run,
    quantities=(
        "positions",
        "cell",
        "forces",
        "temperature",
        "energy_pot",
        "energy_tot",
        "pressure",
    ),
):
    run_str_rendered = Template(run_str).render(run=run)
    lmp_instance.interactive_lib_command(run_str_rendered)
    interactive_getter_dict = {
        "positions": lmp_instance.interactive_positions_getter,
        "cell": lmp_instance.interactive_cells_getter,
        "forces": lmp_instance.interactive_forces_getter,
        "temperature": lmp_instance.interactive_temperatures_getter,
        "energy_pot": lmp_instance.interactive_energy_pot_getter,
        "energy_tot": lmp_instance.interactive_energy_tot_getter,
        "pressure": lmp_instance.interactive_pressures_getter,
    }
    return {q: interactive_getter_dict[q]() for q in quantities}


def lammps_calc_md(
    lmp_instance,
    run_str,
    run,
    thermo,
    quantities=(
        "positions",
        "cell",
        "forces",
        "temperature",
        "energy_pot",
        "energy_tot",
        "pressure",
    ),
):
    results_lst = [
        lammps_calc_md_step(
            lmp_instance=lmp_instance,
            run_str=run_str,
            run=thermo,
            quantities=quantities,
        )
        for _ in range(run // thermo)
    ]
    return {q: np.array([d[q] for d in results_lst]) for q in quantities}


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
    for temp in temperature_lst:
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
    return temperature_md_lst, volume_md_lst


def lammps_shutdown(lmp_instance, close_instance=True):
    lmp_instance.interactive_lib_command("clear")
    if close_instance:
        lmp_instance.close()
