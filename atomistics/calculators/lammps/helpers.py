from __future__ import annotations

import numpy as np
import pandas
from ase.atoms import Atoms
from jinja2 import Template
from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.lammps.potential import validate_potential_dataframe
from atomistics.shared.output import OutputMolecularDynamics, OutputThermalExpansion
from atomistics.shared.thermal_expansion import get_thermal_expansion_output
from atomistics.shared.tqdm_iterator import get_tqdm_iterator


def lammps_run(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    input_template=None,
    lmp=None,
    **kwargs,
):
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
        for line in input_template.split("\n"):
            lmp.interactive_lib_command(line)

    return lmp


def lammps_calc_md_step(
    lmp_instance,
    run_str: str,
    run: int,
    output_keys=OutputMolecularDynamics.keys(),
):
    run_str_rendered = Template(run_str).render(run=run)
    lmp_instance.interactive_lib_command(run_str_rendered)
    return OutputMolecularDynamics(
        positions=lmp_instance.interactive_positions_getter,
        cell=lmp_instance.interactive_cells_getter,
        forces=lmp_instance.interactive_forces_getter,
        temperature=lmp_instance.interactive_temperatures_getter,
        energy_pot=lmp_instance.interactive_energy_pot_getter,
        energy_tot=lmp_instance.interactive_energy_tot_getter,
        pressure=lmp_instance.interactive_pressures_getter,
        velocities=lmp_instance.interactive_velocities_getter,
        volume=lmp_instance.interactive_volume_getter,
    ).get(output_keys=output_keys)


def lammps_calc_md(
    lmp_instance,
    run_str: str,
    run: int,
    thermo: int,
    output_keys=OutputMolecularDynamics.keys(),
):
    results_lst = [
        lammps_calc_md_step(
            lmp_instance=lmp_instance,
            run_str=run_str,
            run=thermo,
            output_keys=output_keys,
        )
        for _ in range(run // thermo)
    ]
    return {q: np.array([d[q] for d in results_lst]) for q in output_keys}


def lammps_thermal_expansion_loop(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    init_str: str,
    run_str: str,
    temperature_lst: list[float],
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Tdamp: float = 0.1,
    Pstart: float = 0.0,
    Pstop: float = 0.0,
    Pdamp: float = 1.0,
    seed: int = 4928459,
    dist: str = "gaussian",
    lmp=None,
    output_keys=OutputThermalExpansion.keys(),
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
        for line in run_str_rendered.split("\n"):
            lmp_instance.interactive_lib_command(line)
        volume_md_lst.append(lmp_instance.interactive_volume_getter())
        temperature_md_lst.append(lmp_instance.interactive_temperatures_getter())
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return get_thermal_expansion_output(
        temperatures_lst=temperature_md_lst,
        volumes_lst=volume_md_lst,
        output_keys=output_keys,
    )


def lammps_shutdown(lmp_instance, close_instance: bool = True):
    lmp_instance.interactive_lib_command("clear")
    if close_instance:
        lmp_instance.close()
