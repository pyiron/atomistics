# pyiron_lammps

## Disclaimer
The `pyiron_lammps` package is under development. 

## Idea
* The `pyiron_lammps` package is designed to calculate concentration dependent material properties, like the elastic constants, energy-volume curves and phonons for interatomic potentials. 
* It uses `mpi4py` for parallel execution, does not write any files to the file system and does not use any database. With this combination is it one order of magnitude faster than competing software packages for evaluating material properties for interatomic potentials. 
* In contrast to all other `pyiron_*` packages, `pyiron_lammps` does not depent on `pyiron_base`. 

## Example
```
import os
from ase.build import bulk
from pyiron_lammps import PyironLammpsLibrary, ase_to_pyiron, view_potentials, settings

structure = ase_to_pyiron(bulk("Al", cubic=True).repeat([3,3,3]))
lmp = PyironLammpsLibrary()

lammps_input_template = """\
fix ensemble all nve
variable thermotime equal 100
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo ${thermotime}
run 0"""

df_pot = view_potentials(structure=structure)
df_pot_selected = df_pot[df_pot.Name=='2005--Mendelev-M-I--Al-Fe--LAMMPS--ipr1'].iloc[0]

lmp.interactive_structure_setter(
    structure=structure,
    units="metal",
    dimension=3,
    boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
    atom_style="atomic",
    el_eam_lst=df_pot_selected.Species,
    calc_md=False,
)

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
    
potential_commands = get_potential(df_pot_select=df_pot_selected)

for c in potential_commands:
    lmp.interactive_lib_command(c)
    
for l in lammps_input_template.split("\n"):
    lmp.interactive_lib_command(l)
    
print(lmp.interactive_energy_tot_getter())

lmp.close()
```

## Limitations
* It would be great to use the ASE atoms directly. 

## License and Acknowledgments
`pyiron_base` is licensed under the BSD license.

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
