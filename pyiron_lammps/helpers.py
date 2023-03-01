import os
from ase.build import bulk
from pyiron_lammps.potential import view_potentials
from pyiron_lammps.sqs import get_sqs_structures
from pyiron_lammps.wrapper import PyironLammpsLibrary


def update_potential_paths(df_pot, resource_path):
    config_lst = []
    for row in df_pot.itertuples():
        potential_file_lst = row.Filename
        potential_file_path_lst = [
            os.path.join(resource_path, f) for f in potential_file_lst
        ]
        potential_dict = {os.path.basename(f): f for f in potential_file_path_lst}
        potential_commands = []
        for l in row.Config:
            l = l.replace("\n", "")
            for key, value in potential_dict.items():
                l = l.replace(key, value)
            potential_commands.append(l)
        config_lst.append(potential_commands)
    df_pot["Config"] = config_lst
    return df_pot


def generate_sqs_structure(structure_template, element_lst, count_lst):
    structures, sro_breakdown, num_iterations, cycle_time = get_sqs_structures(
        structure=structure_template,
        mole_fractions={
            el: c / len(structure_template) for el, c in zip(element_lst, count_lst)
        },
    )
    return structures


def get_lammps_engine(
    working_directory=None,
    cores=1,
    comm=None,
    logger=None,
    log_file=None,
    library=None,
    diable_log_file=True,
):
    return PyironLammpsLibrary(
        working_directory=working_directory,
        cores=cores,
        comm=comm,
        logger=logger,
        log_file=log_file,
        library=library,
        diable_log_file=diable_log_file,
    )


def get_ase_bulk(*args, **kwargs):
    return bulk(*args, **kwargs)


def get_potential_dataframe(structure, resource_path):
    return update_potential_paths(
        df_pot=view_potentials(structure=structure, resource_path=resource_path),
        resource_path=resource_path,
    )
