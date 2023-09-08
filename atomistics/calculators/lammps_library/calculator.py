from pylammpsmpi import LammpsASELibrary


lammps_input_template = """\
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo 100
run 0"""


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


def get_potential_energy_from_lammps(structure, lmp, potential_dataframe):
    _run_simulation(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=lammps_input_template,
        lmp=lmp,
    )
    energy = lmp.interactive_energy_tot_getter()
    lmp.interactive_lib_command("clear")
    return energy


def get_forces_from_lammps(structure, lmp, potential_dataframe):
    _run_simulation(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=lammps_input_template,
        lmp=lmp,
    )
    forces = lmp.interactive_forces_getter()
    lmp.interactive_lib_command("clear")
    return forces


def get_lammps_engine(
    working_directory=None,
    cores=1,
    comm=None,
    logger=None,
    log_file=None,
    library=None,
    diable_log_file=True,
):
    return LammpsASELibrary(
        working_directory=working_directory,
        cores=cores,
        comm=comm,
        logger=logger,
        log_file=log_file,
        library=library,
        diable_log_file=diable_log_file,
    )


def evaluate_with_lammps(
    task_dict,
    potential_dataframe,
    working_directory=None,
    cores=1,
    comm=None,
    logger=None,
    log_file=None,
    library=None,
    diable_log_file=True,
):
    result_dict = {}
    lmp = LammpsASELibrary(
        working_directory=working_directory,
        cores=cores,
        comm=comm,
        logger=logger,
        log_file=log_file,
        library=library,
        diable_log_file=diable_log_file,
    )
    if "calc_energy" in task_dict.keys():
        result_dict["energy"] = {
            k: get_potential_energy_from_lammps(
                structure=v,
                lmp=lmp,
                potential_dataframe=potential_dataframe,
            )
            for k, v in task_dict["calc_energy"].items()
        }
    if "calc_forces" in task_dict.keys():
        result_dict["forces"] = {
            k: get_forces_from_lammps(
                structure=v, lmp=lmp, potential_dataframe=potential_dataframe
            )
            for k, v in task_dict["calc_forces"].items()
        }
    lmp.close()
    return result_dict
