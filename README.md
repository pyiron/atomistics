# pyiron_lammps

## Disclaimer
The `pyiron_lammps` package is under development. 

## Idea
* The `pyiron_lammps` package is designed to calculate concentration dependent material properties, like the elastic constants, energy-volume curves and phonons for interatomic potentials. 
* It uses `mpi4py` for parallel execution, does not write any files to the file system and does not use any database. With this combination is it one order of magnitude faster than competing software packages for evaluating material properties for interatomic potentials. 
* In contrast to all other `pyiron_*` packages, `pyiron_lammps` does not depent on `pyiron_base`. 

## Example
```python
import os
from ase.build import bulk
from pyiron_lammps import PyironLammpsLibrary, ase_to_pyiron, view_potentials, settings, get_sqs_structures, ElasticMatrixCalculator

def get_potential(df_pot_select):
    potential_file_lst=df_pot_select["Filename"]
    potential_path = [p for p in settings.resource_paths if "iprpy" in p][-1]
    potential_file_path_lst = [os.path.join(potential_path, f) for f in potential_file_lst]
    potential_dict = {os.path.basename(f): f for f in potential_file_path_lst}
    potential_commands = []
    for l in df_pot_select["Config"]:
        l = l.replace("\n", "")
        for key, value in potential_dict.items():
            l = l.replace(key, value)
        potential_commands.append(l)
    return potential_commands

# Generate SQS structure
structure_template = ase_to_pyiron(bulk("Al", cubic=True).repeat([3,3,3]))
element_lst = ["Fe", "Ni", "Cr", "Co", "Cu"]
count_lst = [22, 22, 22, 21, 21]
structures, sro_breakdown, num_iterations, cycle_time = get_sqs_structures(
    structure=structure_template,
    mole_fractions={el: c/len(structure_template) for el, c in zip(element_lst, count_lst)},
)
structure = structures[0]

# Minimize structure 
potential = '2021--Deluigi-O-R--Fe-Ni-Cr-Co-Cu--LAMMPS--ipr1'
lmp = PyironLammpsLibrary()
lammps_input_template_static = """\
fix ensemble all nve
variable thermotime equal 100
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo ${thermotime}
run 0"""

line_lst = []
for l in lammps_input_template_static.split("\n"):
    if "fix ensemble" in l:
        line_lst.append("fix ensemble all box/relax iso 0.0")
    elif "run" in l:
        line_lst.append("min_style cg")
        line_lst.append("minimize 0.0 0.0001 100000 10000000")
    else:
        line_lst.append(l)
lammps_input_template_minimize_cell = "\n".join(line_lst)

df_pot = view_potentials(structure=structure)
df_pot_selected = df_pot[df_pot.Name==potential].iloc[0]

lmp.interactive_structure_setter(
    structure=structure,
    units="metal",
    dimension=3,
    boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
    atom_style="atomic",
    el_eam_lst=df_pot_selected.Species,
    calc_md=False,
)

potential_commands = get_potential(df_pot_select=df_pot_selected)

for c in potential_commands:
    lmp.interactive_lib_command(c)

for l in lammps_input_template_minimize_cell.split("\n"):
    lmp.interactive_lib_command(l)
    
structure_copy = structure.copy()
structure_copy.set_cell(lmp.interactive_cells_getter(), scale_atoms=True)
structure_copy.positions = lmp.interactive_positions_getter()

lmp.interactive_lib_command("clear")

# Elastic Constants 
line_lst = []
for l in lammps_input_template_minimize_cell.split("\n"):
    if "fix ensemble" not in l:
        line_lst.append(l)
lammps_input_template_minimize_pos = "\n".join(line_lst)

calculator = ElasticMatrixCalculator(
    basis_ref=structure_copy.copy(),
    num_of_point=5,
    eps_range=0.005,
    sqrt_eta=True,
    fit_order=2
)

structure_dict = calculator.generate_structures()

energy_tot_lst = {}
for key, struct in structure_dict.items():
    lmp.interactive_structure_setter(
        structure=struct,
        units="metal",
        dimension=3,
        boundary=" ".join(["p" if coord else "f" for coord in struct.pbc]),
        atom_style="atomic",
        el_eam_lst=df_pot_selected.Species,
        calc_md=False,
    )
    for c in potential_commands:
        lmp.interactive_lib_command(c)
    for l in lammps_input_template_minimize_pos.split("\n"):
        lmp.interactive_lib_command(l)
    energy_tot_lst[key] = lmp.interactive_energy_tot_getter()
    lmp.interactive_lib_command("clear")
    
calculator.analyse_structures(energy_tot_lst)

print(calculator._data['C'])

lmp.close()
```

## Limitations / Next steps
* [ ] Use the ASE atoms directly.
* [X] Add elastic constant, energy volume curve and phonon calculations. 
* [X] Use LAMMPS as engine so multiple GenericMasters can use the same LAMMPS instance. 

## License and Acknowledgments
`pyiron_lammps` is licensed under the BSD license.

If you use pyiron in your scientific work, [please consider citing](http://www.sciencedirect.com/science/article/pii/S0927025618304786):

```
@article{pyiron-paper,
    title = {pyiron: An integrated development environment for computational materials science},
    journal = {Computational Materials Science},
    volume = {163},
    pages = {24 - 36},
    year = {2019},
    issn = {0927-0256},
    doi = {https://doi.org/10.1016/j.commatsci.2018.07.043},
    url = {http://www.sciencedirect.com/science/article/pii/S0927025618304786},
    author = {Jan Janssen and Sudarsan Surendralal and Yury Lysogorskiy and Mira Todorova and Tilmann Hickel and Ralf Drautz and Jörg Neugebauer},
    keywords = {Modelling workflow, Integrated development environment, Complex simulation protocols},
}
```
