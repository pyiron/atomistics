from pyiron_lammps.decorator import calculation
from pyiron_lammps.elastic import ElasticMatrixCalculator


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


@calculation
def calculate_elastic_constants_with_minimization(
    lmp,
    structure,
    potential_dataframe,
    num_of_point=5,
    eps_range=0.005,
    sqrt_eta=True,
    fit_order=2,
):
    structure_optimized = optimize_structure(
        lmp=lmp, structure=structure, potential_dataframe=potential_dataframe
    )
    return calculate_elastic_constants(
        lmp=lmp,
        structure=structure_optimized,
        potential_dataframe=potential_dataframe,
        num_of_point=num_of_point,
        eps_range=eps_range,
        sqrt_eta=sqrt_eta,
        fit_order=fit_order,
    )
