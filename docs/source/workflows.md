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
The elastic constants and elastic moduli can be calculated using the `ElasticMatrixWorkflow`: 
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
print(fit_dict)
```
The `ElasticMatrixWorkflow` takes an `ase.atoms.Atoms` object as `structure` input as well as the number of points 
`num_of_point` for each compression direction. Depending on the symmetry of the input `structure` the number of 
calculations required to calculate the elastic matrix changes. The compression and elongation range is defined by the
`eps_range` parameter. Furthermore, `sqrt_eta` and `fit_order` describe how the change in energy over compression and
elongation is fitted to calculate the resulting pressure. 

## Energy Volume Curve
The `EnergyVolumeCurveWorkflow` can be used to calculate the equilibrium properties: equilibrium volume, equilibrium 
energy, equilibrium bulk modulus and the pressure derivative of the equilibrium bulk modulus. 
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
print(fit_dict)
```
The input parameters for the `EnergyVolumeCurveWorkflow` in addition to the `ase.atoms.Atoms` object defined 
as `structure` are: 

* `num_points` the number of strains to calculate energies and volumes.  
* `fit_type` the type of the fit which should be used to calculate the equilibrium properties. This can either be a 
  `polynomial` fit or a specific equation of state like the Birch equation (`birch`), the Birch-Murnaghan equation 
  (`birchmurnaghan`) the Murnaghan equation (`murnaghan`), the Pourier Tarnatola eqaution (`pouriertarantola`) or the
  Vinet equation (`vinet`).  
* `fit_order` for the `polynomial` fit type the order of the polynomial can be set, for the other fit types this 
  parameter is ignored. 
* `vol_range` specifies the amount of compression and elongation to be applied relative to the absolute volume. 
* `axes` specifies the axes which are compressed, typically a uniform compression is applied. 
* `strains` specifies the strains directly rather than deriving them from the range of volume compression `vol_range`. 

Beyond calculating the equilibrium properties the `EnergyVolumeCurveWorkflow` can also be used to calculate the thermal
expansion using the [Moruzzi, V. L. et al.](https://link.aps.org/doi/10.1103/PhysRevB.37.790)  model: 
```
temperatures, volumes = workflow.get_thermal_expansion(
    output_dict=result_dict, 
    t_min=1, 
    t_max=1500, 
    t_step=50, 
    temperatures=None,
)
```
The [Moruzzi, V. L. et al.](https://link.aps.org/doi/10.1103/PhysRevB.37.790)  model is a quantum mechanical approximation, so the equilibrium volume at 0K is not
the same as the equilibrium volume calculated by fitting the equation of state. 

## Molecular Dynamics 
Just like the structure optimization also the molecular dynamics calculation can either be implemented inside the
simulation code or in the `atomistics` package. The latter has the advantage that it is the same implementation for all
different simulation codes, while the prior has the advantage that it is usually faster and computationally more 
efficient.

### Implemented in the Simulation Code 
The [LAMMPS](https://lammps.org/) simulation code implements a wide range of different simulation workflows, this 
includes molecular dynamics. In the `atomistics` package these can be directly accessed via the python interface. 

#### Langevin Thermostat
The Langevin thermostat is currently the only thermostat which is available as both a stand-alone python interface and
an integrated interface inside the [LAMMPS](https://lammps.org/) simulation code. The latter is introduced here:
```
from ase.build import bulk
from atomistics.calculators import (
    calc_molecular_dynamics_langevin_with_lammps, 
    get_potential_by_name,
)

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
result_dict = calc_molecular_dynamics_langevin_with_lammps(
    structure=bulk("Al", cubic=True).repeat([10, 10, 10]),
    potential_dataframe=potential_dataframe,
    Tstart=100,
    Tstop=100,
    Tdamp=0.1,
    run=100,
    thermo=10,
    timestep=0.001,
    seed=4928459,
    dist="gaussian",
    quantities=("positions", "cell", "forces", "temperature", "energy_pot", "energy_tot", "pressure", "velocities"),
)
```
In addition to the typical LAMMPS input parameters like the atomistic structure `structure` as `ase.atoms.Atoms` object
and the `pandas.DataFrame` for the interatomic potential `potential_dataframe` are: 

* `Tstart` start temperature 
* `Tstop` end temperature
* `Tdamp` temperature damping parameter 
* `run` number of molecular dynamics steps to be executed during one temperature step
* `thermo` refresh rate for the thermo dynamic properties, this should typically be the same as the number of molecular
  dynamics steps. 
* `timestep` time step - typically 1fs defined as `0.001`.
* `seed` random seed for the molecular dynamics 
* `dist` initial velocity distribution 
* `lmp` Lammps library instance as `pylammpsmpi.LammpsASELibrary` object 
* `quantities` the quantities which are extracted from the molecular dynamics simulation

#### Nose Hoover Thermostat
Canonical ensemble (nvt) - volume and temperature constraints molecular dynamics:
```
from ase.build import bulk
from atomistics.calculators import (
    calc_molecular_dynamics_nvt_with_lammps, 
    get_potential_by_name,
)

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
result_dict = calc_molecular_dynamics_nvt_with_lammps(
    structure=bulk("Al", cubic=True).repeat([10, 10, 10]),
    potential_dataframe=potential_dataframe,
    Tstart=100,
    Tstop=100,
    Tdamp=0.1,
    run=100,
    thermo=10,
    timestep=0.001,
    seed=4928459,
    dist="gaussian",
    quantities=("positions", "cell", "forces", "temperature", "energy_pot", "energy_tot", "pressure", "velocities"),
)
```
In addition to the typical LAMMPS input parameters like the atomistic structure `structure` as `ase.atoms.Atoms` object
and the `pandas.DataFrame` for the interatomic potential `potential_dataframe` are: 

* `Tstart` start temperature 
* `Tstop` end temperature
* `Tdamp` temperature damping parameter 
* `run` number of molecular dynamics steps to be executed during one temperature step
* `thermo` refresh rate for the thermo dynamic properties, this should typically be the same as the number of molecular
  dynamics steps. 
* `timestep` time step - typically 1fs defined as `0.001`.
* `seed` random seed for the molecular dynamics 
* `dist` initial velocity distribution 
* `lmp` Lammps library instance as `pylammpsmpi.LammpsASELibrary` object 
* `quantities` the quantities which are extracted from the molecular dynamics simulation

Isothermal-isobaric ensemble (npt) - pressure and temperature constraints molecular dynamics:
```
from ase.build import bulk
from atomistics.calculators import (
    calc_molecular_dynamics_npt_with_lammps, 
    get_potential_by_name,
)

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
result_dict = calc_molecular_dynamics_npt_with_lammps(
    structure=bulk("Al", cubic=True).repeat([10, 10, 10]),
    potential_dataframe=potential_dataframe,
    Tstart=100,
    Tstop=100,
    Tdamp=0.1,
    run=100,
    thermo=100,
    timestep=0.001,
    Pstart=0.0,
    Pstop=0.0,
    Pdamp=1.0,
    seed=4928459,
    dist="gaussian",
    quantities=("positions", "cell", "forces", "temperature", "energy_pot", "energy_tot", "pressure", "velocities"),
)
```
The input parameters for the isothermal-isobaric ensemble (npt) are the same as for the canonical ensemble (nvt) plus:

* `Pstart` start pressure 
* `Pstop` end pressure 
* `Pdamp` pressure damping parameter 

Isenthalpic ensemble (nph) - pressure and helmholtz-energy constraints molecular dynamics:
```
from ase.build import bulk
from atomistics.calculators import (
    calc_molecular_dynamics_nph_with_lammps, 
    get_potential_by_name,
)

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
result_dict = calc_molecular_dynamics_nph_with_lammps(
    structure=bulk("Al", cubic=True).repeat([10, 10, 10]),
    potential_dataframe=potential_dataframe,
    run=100,
    thermo=100,
    timestep=0.001,
    Tstart=100,
    Pstart=0.0,
    Pstop=0.0,
    Pdamp=1.0,
    seed=4928459,
    dist="gaussian",
    quantities=("positions", "cell", "forces", "temperature", "energy_pot", "energy_tot", "pressure", "velocities"),
)
```

#### Thermal Expansion
One example of a molecular dynamics calculation with the LAMMPS simulation code is the calculation of the thermal 
expansion: 
```
from ase.build import bulk
from atomistics.calculators import (
    calc_molecular_dynamics_thermal_expansion_with_lammps,  
    get_potential_by_name,
)

potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
temperatures_md, volumes_md = calc_molecular_dynamics_thermal_expansion_with_lammps(
    structure=bulk("Al", cubic=True).repeat([10, 10, 10]),
    potential_dataframe=potential_dataframe,
    Tstart=100,
    Tstop=1000,
    Tstep=100,
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
In addition to the typical LAMMPS input parameters like the atomistic structure `structure` as `ase.atoms.Atoms` object
and the `pandas.DataFrame` for the interatomic potential `potential_dataframe` are: 

* `Tstart` start temperature 
* `Tstop` end temperature 
* `Tstep` temperature step 
* `Tdamp` temperature damping parameter 
* `run` number of molecular dynamics steps to be executed during one temperature step
* `thermo` refresh rate for the thermo dynamic properties, this should typically be the same as the number of molecular
  dynamics steps. 
* `timestep` time step - typically 1fs defined as `0.001`.
* `Pstart` start pressure 
* `Pstop` end pressure 
* `Pdamp` pressure damping parameter 
* `seed` random seed for the molecular dynamics 
* `dist` initial velocity distribution 
* `lmp` Lammps library instance as `pylammpsmpi.LammpsASELibrary` object 

These input parameters are based on the LAMMPS fix `nvt/npt`, you can read more about the specific implementation on the
[LAMMPS website](https://docs.lammps.org/fix_nh.html).

#### Phonons from Molecular Dynamics
The softening of the phonon modes is calculated for Silicon using the [Tersoff interatomic potential](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.38.9902) 
which is available via the [NIST potentials repository](https://www.ctcms.nist.gov/potentials/entry/1988--Tersoff-J--Si-c/). 
Silicon is chosen based on its diamond crystal lattice which requires less calculation than the face centered cubic (fcc)
crystal of Aluminium. The simulation workflow consists of three distinct steps:

* Starting with the optimization of the equilibrium structure. 
* Followed by the calculation of the 0K phonon spectrum. 
* Finally, the finite temperature phonon spectrum is calculated using molecular dynamics. 

The finite temperature phonon spectrum is calculated using the [DynaPhoPy](https://abelcarreras.github.io/DynaPhoPy/)
package, which is integrated inside the `atomistics` package. As a prerequisite the dependencies, imported and the bulk 
silicon diamond structure is created and the Tersoff interatomic potential is loaded: 
```
from ase.build import bulk
from atomistics.calculators import (
    calc_molecular_dynamics_phonons_with_lammps,
    evaluate_with_lammps, 
)
from atomistics.workflows import optimize_positions_and_volume, PhonopyWorkflow
from dynaphopy import Quasiparticle
import pandas
from phonopy.units import VaspToTHz
import spglib

structure_bulk = bulk("Si", cubic=True)
potential_dataframe = get_potential_by_name(
    potential_name='1988--Tersoff-J--Si-c--LAMMPS--ipr1'
)
```

The first step is optimizing the Silicon diamond structure to match the lattice specifications implemented in the Tersoff 
interatomic potential:
```
task_dict = optimize_positions_and_volume(structure=structure_bulk)
result_dict = evaluate_with_lammps(
    task_dict=task_dict,
    potential_dataframe=potential_dataframe,
)
structure_ase = result_dict["structure_with_optimized_positions_and_volume"]
```

As a second step the 0K phonons are calculated using the `PhonopyWorkflow` which is explained in more detail below in 
the section on [Phonons](https://atomistics.readthedocs.io/en/latest/workflows.html#phonons). 
```
cell = (structure_ase.cell.array, structure_ase.get_scaled_positions(), structure_ase.numbers)
primitive_matrix = spglib.standardize_cell(cell=cell, to_primitive=True)[0] / structure_ase.get_volume() ** (1/3)
workflow = PhonopyWorkflow(
    structure=structure_ase,
    interaction_range=10,
    factor=VaspToTHz,
    displacement=0.01,
    dos_mesh=20,
    primitive_matrix=primitive_matrix,
    number_of_snapshots=None,
)
task_dict = workflow.generate_structures()
result_dict = evaluate_with_lammps(
    task_dict=task_dict,
    potential_dataframe=potential_dataframe,
)
workflow.analyse_structures(output_dict=result_dict)
```

The calcualtion of the finite temperature phonons starts by computing the molecular dynamics trajectory using the 
`calc_molecular_dynamics_phonons_with_lammps()` function. This function is internally linked to [DynaPhoPy](https://abelcarreras.github.io/DynaPhoPy/)
to return an `dynaphopy.dynamics.Dynamics` object: 
```
trajectory = calc_molecular_dynamics_phonons_with_lammps(
    structure_ase=structure_ase,
    potential_dataframe=potential_dataframe,
    force_constants=workflow.phonopy.get_force_constants(), 
    phonopy_unitcell=workflow.phonopy.get_unitcell(),
    phonopy_primitive_matrix=workflow.phonopy.get_primitive_matrix(),
    phonopy_supercell_matrix=workflow.phonopy.get_supercell_matrix(),
    total_time=2,       # ps
    time_step=0.001,    # ps
    relaxation_time=5,  # ps
    silent=True,
    supercell=[2, 2, 2],
    memmap=False,
    velocity_only=True,
    temperature=100,
)
```
When a total of 2 picoseconds is selected to compute the finite temperature phonons with a timestep of 1 femto second
then this results in a total of 2000 molecular dynamics steps. While more molecular dynamics steps result in more precise
predictions they also require more computational resources. 

The postprocessing is executed using the [DynaPhoPy](https://abelcarreras.github.io/DynaPhoPy/) package: 
```
calculation = Quasiparticle(trajectory)
calculation.select_power_spectra_algorithm(2)  # select FFT algorithm
calculation.get_renormalized_phonon_dispersion_bands()
renormalized_force_constants = calculation.get_renormalized_force_constants().get_array()
renormalized_force_constants
```
It calculates the re-normalized force constants which can then be used to calculate the finite temperature properties. 

In addition the [DynaPhoPy](https://abelcarreras.github.io/DynaPhoPy/) package can be used to directly compare the 
finite temperature phonon spectrum with the 0K phonon spectrum calulated with the finite displacement method: 
```
calculation.plot_renormalized_phonon_dispersion_bands()
```
![finite temperature band_structure](../pictures/lammps_md_phonons.png)

### Langevin Thermostat 
In addition to the molecular dynamics implemented in the LAMMPS simulation code, the `atomistics` package also provides
the `LangevinWorkflow` which implements molecular dynamics independent of the specific simulation code. 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps_library, get_potential_by_name
from atomistics.workflows import LangevinWorkflow
from pylammpsmpi import LammpsASELibrary

steps = 300
potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)
workflow = LangevinWorkflow(
    structure=bulk("Al", cubic=True).repeat([2, 2, 2]), 
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
        potential_dataframe=potential_dataframe,
        lmp=lmp,
    )
    eng_pot, eng_kin = workflow.analyse_structures(output_dict=result_dict)
    eng_pot_lst.append(eng_pot)
    eng_kin_lst.append(eng_kin)
lmp.close()
```
The advantage of this implementation is that the user can directly interact with the simulation between the individual
molecular dynamics simulation steps. This provides a lot of flexibility to prototype new simulation methods. The input
parameters of the `LangevinWorkflow` are:

* `structure` the `ase.atoms.Atoms` object which is used as initial structure for the molecular dynamics calculation 
* `temperature` the temperature of the molecular dynamics calculation given in Kelvin
* `overheat_fraction` the over heating fraction of the Langevin thermostat
* `damping_timescale` the damping timescale of the Langevin thermostat 
* `time_step` the time steps of the Langevin thermostat

## Harmonic Approximation 
The harmonic approximation is implemented in two variations, once with constant volume and once including the volume 
expansion at finite temperature also known as quasi-harmonic approximation. Both of these are based on the [phonopy](https://phonopy.github.io/phonopy/)
package. 

### Phonons 
To calculate the phonons at a fixed volume the `PhonopyWorkflow` is used:
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
mesh_dict, dos_dict = workflow.analyse_structures(output_dict=result_dict)
```
The `PhonopyWorkflow` takes the following inputs: 

* `structure` the `ase.atoms.Atoms` object to calculate the phonon spectrum
* `interaction_range` the cutoff radius to consider for identifying the interaction between the atoms
* `factor` conversion factor, typically just `phonopy.units.VaspToTHz` 
* `displacement` displacement to calculate the forces 
* `dos_mesh` mesh for the density of states 
* `primitive_matrix` primitive matrix
* `number_of_snapshots` number of snapshots to calculate

In addition to the phonon properties, the `PhonopyWorkflow` also enables the calculation of thermal properties: 
```
tp_dict = workflow.get_thermal_properties(
    t_min=1, 
    t_max=1500, 
    t_step=50, 
    temperatures=None,
    cutoff_frequency=None,
    pretend_real=False,
    band_indices=None,
    is_projection=False,
)
print(tp_dict)
```
The calculation of the thermal properties takes additional inputs: 

* `t_min` minimum temperature
* `t_max` maximum temperature
* `t_step` temperature step 
* `temperatures` alternative to `t_min`, `t_max` and `t_step` the array of temperatures can be defined directly
* `cutoff_frequency` cutoff frequency to exclude the contributions of frequencies below a certain cut off
* `pretend_real` use the absolute values of the phonon frequencies
* `band_indices` select bands based on their indices 
* `is_projection` multiplies the squared eigenvectors - not recommended

Furthermore, also the dynamical matrix can be directly calculated with the `PhonopyWorkflow`:
```
mat = workflow.get_dynamical_matrix()
```

Or alternatively the hesse matrix: 
```
mat = workflow.get_hesse_matrix()
```

Finally, also the function to calculate the band structure is directly available on the `PhonopyWorkflow`: 
```
band_structure = workflow.get_band_structure(
    npoints=101, 
    with_eigenvectors=False, 
    with_group_velocities=False
)
```

This band structure can also be visualised using the built-in plotting function: 
```
workflow.plot_band_structure()
```
![band_structure](../pictures/phonon_bands_al.png)

Just like the desnsity of states which can be plotted using: 
```
workflow.plot_dos()
```
![density of states](../pictures/phonon_dos_al.png)

### Quasi-harmonic Approximation 
To include the volume expansion with finite temperature the `atomistics` package implements the `QuasiHarmonicWorkflow`:
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
The `QuasiHarmonicWorkflow` is a combination of the `EnergyVolumeCurveWorkflow` and the `PhonopyWorkflow`. Consequently, 
the inputs are a superset of the inputs of these two workflows. 

Based on the `QuasiHarmonicWorkflow` the thermal expansion can be calculated:
```
temperatures, volumes = workflow.get_thermal_expansion(
    output_dict=result_dict, 
    t_min=1, 
    t_max=1500, 
    t_step=50, 
    temperatures=None,
    cutoff_frequency=None,
    pretend_real=False,
    band_indices=None,
    is_projection=False,
    quantum_mechanical=True,
)
```
This requires the same inputs as the calculation of the thermal properties `get_thermal_properties()` with the 
`PhonopyWorkflow`. The additional parameter `quantum_mechanical` specifies whether the classical harmonic oscillator or 
the quantum mechanical harmonic oscillator is used to calculate the free energy. 

## Structure Optimization 
In analogy to the molecular dynamics calculation also the structure optimization could in principle be defined inside 
the simulation code or on the python level. Still currently the `atomistics` package only supports the structure 
optimization defined inside the simulation codes. 

### Volume and Positions 
To optimize both the volume of the supercell as well as the positions inside the supercell the `atomistics` package
implements the `optimize_positions_and_volume()` workflow:
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
The result is the optimized atomistic structure as part of the result dictionary. 

### Positions 
The optimization of the positions inside the supercell without the optimization of the supercell volume is possible with
the `optimize_positions()` workflow:
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
The result is the optimized atomistic structure as part of the result dictionary. 