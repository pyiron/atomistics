import os
from ase.build import bulk
from pyiron_lammps.decorator import calculation
from pyiron_lammps.elastic import ElasticMatrixCalculator
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


@calculation
def optimize_structure(lmp, structure, potential_dataframe):
    lammps_input_template_minimize_cell = """\
fix ensemble all box/relax iso 0.0
variable thermotime equal 100
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo ${thermotime}
min_style cg
minimize 0.0 0.0001 100000 10000000"""

    lmp = _run_simulation(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=lammps_input_template_minimize_cell,
        lmp=lmp,
    )

    # get final structure
    structure_copy = structure.copy()
    structure_copy.set_cell(lmp.interactive_cells_getter(), scale_atoms=True)
    structure_copy.positions = lmp.interactive_positions_getter()

    # clean memory
    lmp.interactive_lib_command("clear")
    return structure_copy


@calculation
def calculate_elastic_constants(
    lmp,
    structure,
    potential_dataframe,
    num_of_point=5,
    eps_range=0.005,
    sqrt_eta=True,
    fit_order=2,
):
    lammps_input_template_minimize_pos = """\
variable thermotime equal 100
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo ${thermotime}
min_style cg
minimize 0.0 0.0001 100000 10000000"""

    # Generate structures
    calculator = ElasticMatrixCalculator(
        basis_ref=structure.copy(),
        num_of_point=num_of_point,
        eps_range=eps_range,
        sqrt_eta=sqrt_eta,
        fit_order=fit_order,
    )
    structure_dict = calculator.generate_structures()

    # run calculation
    energy_tot_lst = {}
    for key, struct in structure_dict.items():
        lmp = _run_simulation(
            lmp=lmp,
            structure=struct,
            potential_dataframe=potential_dataframe,
            input_template=lammps_input_template_minimize_pos,
        )
        energy_tot_lst[key] = lmp.interactive_energy_tot_getter()
        lmp.interactive_lib_command("clear")

    # fit
    calculator.analyse_structures(energy_tot_lst)
    return calculator._data["C"]


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
