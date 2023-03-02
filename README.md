# pyiron_lammps

## Disclaimer
The `pyiron_lammps` package is under development. 

## Idea
* The `pyiron_lammps` package is designed to calculate concentration dependent material properties, like the elastic constants, energy-volume curves and phonons for interatomic potentials. 
* It uses `mpi4py` for parallel execution, does not write any files to the file system and does not use any database. With this combination is it one order of magnitude faster than competing software packages for evaluating material properties for interatomic potentials. 
* In contrast to all other `pyiron_*` packages, `pyiron_lammps` does not depent on `pyiron_base`. Overall the dependencies are designed to be minimal. 

## Example
```python
import pyiron_lammps as pyr

# Generate SQS Structure
structure = pyr.generate_sqs_structure(
    structure_template=pyr.get_ase_bulk("Al", cubic=True).repeat([3,3,3]), 
    element_lst=["Fe", "Ni", "Cr", "Co", "Cu"], 
    count_lst=[22, 22, 22, 21, 21]
)[0]

# Select Potential
potential = '2021--Deluigi-O-R--Fe-Ni-Cr-Co-Cu--LAMMPS--ipr1'
df_pot = pyr.get_potential_dataframe(
    structure=structure, 
    resource_path="/Users/janssen/mambaforge/share/iprpy"
)
df_pot_selected = df_pot[df_pot.Name==potential].iloc[0]

# Optimize Structure
structure_opt = pyr.optimize_structure(
    structure=structure, 
    potential_dataframe=df_pot_selected
)

# Calculate Elastic Constants
elastic_matrix = pyr.calculate_elastic_constants(
    structure=structure_opt, 
    potential_dataframe=df_pot_selected, 
    num_of_point=5, 
    eps_range=0.005, 
    sqrt_eta=True, 
    fit_order=2
)
print(elastic_matrix)
```

## Features
* `generate_sqs_structure` - generate an SQS structure using `sqsgenerator`.
* `get_ase_bulk` - create an `ase.build.bulk` structure.
* `get_lammps_engine` - create an `LAMMPS` instance, these instances can then be shared between multiple serial calculation.
* `get_potential_dataframe` - load dataframe of suitable interatomic potentials for the selected atomistic structure from the NIST database.
* `optimize_structure` - optimize the `cell` and the `positions` of a given structure, while maintaining the cell shape.
* `calculate_elastic_constants` - calculate the elastic constants.
* `calculate_elastic_constants_with_minimization` - combine the structure optimization and the calculation of the elastic constants.
* `optimize_structure_parallel` - optimize a list of atomistic structures all with the same interatomic potential. 
* `calculate_elastic_constants_parallel` - calculate the elastic constants for a list of atomistic structures. 
* `calculate_elastic_constants_with_minimization_parallel` - combine the structure optimization and the calculation of the elastic constants for a list of atomistic structures. 

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
