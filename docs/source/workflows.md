# Workflows
To demonstrate the workflows implemented in the `atomistics` package, the [LAMMPS](https://www.lammps.org/) molecular 
dynamics simulation code is used in the following demonstrations. Still the same `workflows` can also be used with other
simulation codes:
```
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
result_dict = evaluate_with_lammps(
    task_dict={},
    potential_dataframe=potential_dataframe,
)
```
The interatomic potential for Aluminium from Mishin named `1999--Mishin-Y--Al--LAMMPS--ipr1` is used in the evaluation
with [LAMMPS](https://www.lammps.org/) `evaluate_with_lammps()`. 

## Elastic Matrix 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name
from atomistics.workflows import ElasticMatrixWorkflow

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
workflow = ElasticMatrixWorkflow(
    structure=bulk("Al", cubic=True), 
    num_of_point=5, 
    eps_range=0.005, 
    sqrt_eta=True, 
    fit_order=2,
)
task_dict = workflow.generate_structures()
result_dict = evaluate_with_lammps(
    task_dict=task_dict,
    potential_dataframe=potential_dataframe,
)
fit_dict = workflow.analyse_structures(output_dict=result_dict)
```

## Energy Volume Curve
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name
from atomistics.workflows import EnergyVolumeCurveWorkflow

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
workflow = EnergyVolumeCurveWorkflow(
    structure=bulk("Al", cubic=True), 
    num_points=11,
    fit_type="polynomial",
    fit_order=3,
    vol_range=0.05,
    axes=("x", "y", "z"),
    strains=None,
)
task_dict = workflow.generate_structures()
result_dict = evaluate_with_lammps(
    task_dict=task_dict,
    potential_dataframe=potential_dataframe,
)
fit_dict = workflow.analyse_structures(output_dict=result_dict)
```

Calculate thermal expansion
```
temperatures, volumes = workflow.get_thermal_expansion(
    output_dict=result_dict, 
    t_min=1, 
    t_max=1500, 
    t_step=50, 
    temperatures=None,
)
```
## Molecular Dynamics 

### Implemented in Simulation Code 
```
from ase.build import bulk
from atomistics.calculators import (
    calc_molecular_dynamics_thermal_expansion_with_lammps, 
    evaluate_with_lammps, 
    get_potential_by_name,
)

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
temperatures, volumes = calc_molecular_dynamics_thermal_expansion_with_lammps(
    structure=bulk("Al", cubic=True).repeat([10, 10, 10])
    potential_dataframe=potential_dataframe,
    Tstart=15,
    Tstop=1500,
    Tstep=5,
    Tdamp=0.1,
    run=100,
    thermo=100,
    timestep=0.001,
    Pstart=0.0,
    Pstop=0.0,
    Pdamp=1.0,
    seed=4928459,
    dist="gaussian",
    lmp=None,
)
```

### Langevin Thermostat 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name, LammpsASELibrary
from atomistics.workflows import LangevinWorkflow

steps = 300
potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
workflow = LangevinWorkflow(
    structure=bulk("Al", cubic=True).repeat([2, 2, 2], 
    temperature=1000.0,
    overheat_fraction=2.0,
    damping_timescale=100.0,
    time_step=1,
)
lmp = LammpsASELibrary(
    working_directory=None,
    cores=1,
    comm=None,
    logger=None,
    log_file=None,
    library=None,
    diable_log_file=True,
)
eng_pot_lst, eng_kin_lst = [], []
for i in range(steps):
    task_dict = workflow.generate_structures()
    result_dict = evaluate_with_lammps_library(
        task_dict=task_dict,
        potential_dataframe=df_pot_selected,
        lmp=lmp,
    )
    eng_pot, eng_kin = workflow.analyse_structures(output_dict=result_dict)
    eng_pot_lst.append(eng_pot)
    eng_kin_lst.append(eng_kin)
lmp.close()
```

## Harmonic Approximation 

### Phonons 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name
from atomistics.workflows import PhonopyWorkflow
from phonopy.units import VaspToTHz

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
workflow = PhonopyWorkflow(
    structure=bulk("Al", cubic=True), 
    interaction_range=10,
    factor=VaspToTHz,
    displacement=0.01,
    dos_mesh=20,
    primitive_matrix=None,
    number_of_snapshots=None,
)
task_dict = workflow.generate_structures()
result_dict = evaluate_with_lammps(
    task_dict=task_dict,
    potential_dataframe=potential_dataframe,
)
fit_dict = workflow.analyse_structures(output_dict=result_dict)
```

```
tp_dict = workflow.get_thermal_properties(t_min=1, t_max=1500, t_step=50, temperatures=None)
```

```
mat = workflow.get_dynamical_matrix()
```

```
mat = workflow.get_hesse_matrix()
```

```
band_structure = workflow.get_band_structure(npoints=101, with_eigenvectors=False, with_group_velocities=False)
```

Thermal Expansion:
```
temperatures, volumes = workflow.get_thermal_expansion(
    output_dict=output_dict, 
    t_min=1, 
    t_max=1500, 
    t_step=50, 
    temperatures=None,
)
```
Plotting:
```
workflow.plot_band_structure()
```

```
workflow.plot_dos()
```
### Quasi-harmonic Approximation 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name
from atomistics.workflows import QuasiHarmonicWorkflow

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
workflow = QuasiHarmonicWorkflow(
    structure=bulk("Al", cubic=True), 
    num_points=11,
    vol_range=0.05,
    interaction_range=10,
    factor=VaspToTHz,
    displacement=0.01,
    dos_mesh=20,
    primitive_matrix=None,
    number_of_snapshots=None,
)
task_dict = workflow.generate_structures()
result_dict = evaluate_with_lammps(
    task_dict=task_dict,
    potential_dataframe=potential_dataframe,
)
fit_dict = workflow.analyse_structures(output_dict=result_dict)
```

Thermal Expansion:
```
temperatures, volumes = workflow.get_thermal_expansion(
    output_dict=output_dict, 
    t_min=1, 
    t_max=1500, 
    t_step=50, 
    temperatures=None,
)
```

## Structure Optimization 
### Volume and Positions 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name
from atomistics.workflows import optimize_positions_and_volume

structure = bulk("Al", a=4.0, cubic=True)
potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
result_dict = evaluate_with_lammps(
    task_dict=optimize_positions_and_volume(structure=structure),
    potential_dataframe=potential_dataframe,
)
structure_opt = result_dict["structure_with_optimized_positions_and_volume"]
```

### Positions 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name
from atomistics.workflows import optimize_positions

structure = bulk("Al", a=4.0, cubic=True)
potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
result_dict = evaluate_with_lammps(
    task_dict=optimize_positions(structure=structure),
    potential_dataframe=potential_dataframe,
)
structure_opt = result_dict["structure_with_optimized_positions"]
```