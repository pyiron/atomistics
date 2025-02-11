# atomistics
[![Pipeline](https://github.com/pyiron/atomistics/actions/workflows/pipeline.yml/badge.svg)](https://github.com/pyiron/atomistics/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/gh/pyiron/atomistics/graph/badge.svg?token=8LET56AS45)](https://codecov.io/gh/pyiron/atomistics)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyiron/atomistics/HEAD)

The `atomistics` package consists of two primary components. On the one hand it provides interfaces to atomistic
simulation codes - named `calculators`. The supported simulation codes in alphabetical order are:

* [Abinit](https://www.abinit.org) - Plane wave density functional theory
* [EMT](https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html) - Effective medium theory potential
* [GPAW](https://wiki.fysik.dtu.dk/gpaw/) - Density functional theory Python code based on the projector-augmented wave method
* [LAMMPS](https://www.lammps.org) - Molecular Dynamics
* [Quantum Espresso](https://www.quantum-espresso.org) - Integrated suite of Open-Source computer codes for electronic-structure calculations
* [Siesta](https://siesta-project.org) - Electronic structure calculations and ab initio molecular dynamics

For majority of these simulation codes the `atomistics` package use the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)
to interface the underlying C/ C++ and Fortran Codes with the Python programming language. Still this approach limits
the functionality of the simulation code to calculating the energy and forces, so by adding custom interfaces the
`atomistics` package can support built-in features of the simulation code like structure optimization and molecular
dynamics.

On the other hand the `atomistics` package also provides `workflows` to calculate material properties on the atomistic 
scales, these include:

* [Equation of State](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#energy-volume-curve) - to calculate equilibrium properties like the equilibrium energy, equilibrium volume, equilibrium bulk modulus and its pressure derivative.
* [Elastic Matrix](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#elastic-matrix) - to calculate the elastic constants and elastic moduli.
* [Harmonic and Quasi-harmonic Approximation](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#harmonic-approximation) - to calculate the density of states, vibrational free energy and thermal expansion based on the finite displacements method implemented in [phonopy](https://phonopy.github.io/phonopy/).
* [Molecular Dynamics](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#molecular-dynamics) - to calculate finite temperature properties like thermal expansion including the anharmonic contributions.

All these `workflows` can be coupled with all the simulation codes implemented in the `atomistics` package.
In contrast to the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) which provides similar functionality
the focus of the `atomistics` package is not to reimplement existing functionality but rather simplify the process
of coupling existing simulation codes with existing workflows. Here the [phonopy](https://phonopy.github.io/phonopy/)
workflow is a great example to enable the calculation of thermodynamic properties with the harmonic and quasi-harmonic
approximation.

## Example
Use the equation of state to calculate the equilibrium properties like the equilibrium volume, equilibrium energy,
equilibrium bulk modulus and its derivative using the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) simulation code

```python
from ase.build import bulk
from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import EnergyVolumeCurveWorkflow
from gpaw import GPAW, PW

workflow = EnergyVolumeCurveWorkflow(
    structure=bulk("Al", a=4.05, cubic=True),
    num_points=11,
    fit_type='polynomial',
    fit_order=3,
    vol_range=0.05,
    axes=['x', 'y', 'z'],
    strains=None,
)
task_dict = workflow.generate_structures()
print(task_dict)
```
```
>>> {'calc_energy': OrderedDict([
>>>     (0.95, Atoms(symbols='Al4', pbc=True, cell=[3.9813426685908118, 3.9813426685908118, 3.9813426685908118])),
>>>     (0.96, Atoms(symbols='Al4', pbc=True, cell=[3.9952635604153612, 3.9952635604153612, 3.9952635604153612])),
>>>     (0.97, Atoms(symbols='Al4', pbc=True, cell=[4.009088111958974, 4.009088111958974, 4.009088111958974])),
>>>     (0.98, Atoms(symbols='Al4', pbc=True, cell=[4.022817972936038, 4.022817972936038, 4.022817972936038])),
>>>     (0.99, Atoms(symbols='Al4', pbc=True, cell=[4.036454748321015, 4.036454748321015, 4.036454748321015])),
>>>     (1.0, Atoms(symbols='Al4', pbc=True, cell=[4.05, 4.05, 4.05])),
>>>     (1.01, Atoms(symbols='Al4', pbc=True, cell=[4.063455248345461, 4.063455248345461, 4.063455248345461])),
>>>     (1.02, Atoms(symbols='Al4', pbc=True, cell=[4.076821973718458, 4.076821973718458, 4.076821973718458])),
>>>     (1.03, Atoms(symbols='Al4', pbc=True, cell=[4.0901016179023415, 4.0901016179023415, 4.0901016179023415])),
>>>     (1.04, Atoms(symbols='Al4', pbc=True, cell=[4.1032955854717175, 4.1032955854717175, 4.1032955854717175])),
>>>     (1.05, Atoms(symbols='Al4', pbc=True, cell=[4.1164052451001565, 4.1164052451001565, 4.1164052451001565]))
>>> ])}
```

In the first step the `EnergyVolumeCurveWorkflow` object is initialized including all the parameters to generate
the strained structures and afterwards fit the resulting energy volume curve. This allows the user to see all relevant
parameters at one place. After the initialization the function `generate_structures()` is called without any
additional parameters. This function returns the task dictionary `task_dict` which includes the tasks which should
be executed by the calculator. In this case the task is to calculate the energy `calc_energy` of the eleven
generated structures. Each structure is labeled by the ratio of compression or elongation. In the second step the
`task_dict` is evaluate with the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) simulation code using the
`evaluate_with_ase()` function:
```python
result_dict = evaluate_with_ase(
    task_dict=task_dict,
    ase_calculator=GPAW(
        xc="PBE",
        mode=PW(300),
        kpts=(3, 3, 3)
    )
)
print(result_dict)
```
```
>>> {'energy': {
>>>     0.95: -14.895378072824752,
>>>     0.96: -14.910819737657118,
>>>     0.97: -14.922307241122466,
>>>     0.98: -14.930392279321056,
>>>     0.99: -14.935048569964911,
>>>     1.0: -14.936666396364169,
>>>     1.01: -14.935212782128556,
>>>     1.02: -14.931045138839849,
>>>     1.03: -14.924165445706581,
>>>     1.04: -14.914703574005678,
>>>     1.05: -14.902774559134226
>>> }}
```
In analogy to the `task_dict` which defines the tasks to be executed by the simulation code the `result_dict`
summarizes the results of the calculations. In this case the energies calculated for the specific strains. By ordering
both the `task_dict` and the `result_dict` with the same labels, the `EnergyVolumeCurveWorkflow` object
is able to match the calculation results to the corresponding structure. Finally, in the third step the `analyse_structures()`
function takes the `result_dict` as an input and fits the Equation of State with the fitting parameters defined in
the first step:
```python
fit_dict = workflow.analyse_structures(output_dict=result_dict)
print(fit_dict)
```
```
>>> {'poly_fit': array([-9.30297838e-05,  2.19434659e-02, -1.68388816e+00,  2.73605421e+01]),
>>>  'fit_type': 'polynomial',
>>>  'fit_order': 3,
>>>  'volume_eq': 66.44252286131888,
>>>  'energy_eq': -14.93670322204575,
>>>  'bulkmodul_eq': 72.38919826304497,
>>>  'b_prime_eq': 4.45383655040775,
>>>  'least_square_error': 4.432974529908853e-09,
>>>  'volume': [63.10861874999998, 63.77291999999998, ..., 69.75163125000002],
>>>  'energy': [-14.895378072824752, -14.910819737657118, ..., -14.902774559134226]
>>> }
```
As a result the equilibrium parameters are returned plus the parameters of the polynomial and the set of volumes and
energies which were fitted to achieve these results. The important step here is that while the interface between the
first and the second as well as between the second and the third step is clearly defined independent of the specific
workflow, the initial parameters for the workflow to initialize the `EnergyVolumeCurveWorkflow` object as well as
the final output of the `fit_dict` are workflow specific.

## Disclaimer
While we try to develop a stable and reliable software library, the development remains a opensource project under the
BSD 3-Clause License without any warranties:
```
BSD 3-Clause License

Copyright (c) 2023, Jan Janssen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Documentation
* [Installation](https://atomistics.readthedocs.io/en/latest/installation.html)
  * [conda-based Installation](https://atomistics.readthedocs.io/en/latest/installation.html#conda-based-installation)
  * [pypi-based Installation](https://atomistics.readthedocs.io/en/latest/installation.html#pypi-based-installation)
* [Simulation Codes](https://atomistics.readthedocs.io/en/latest/simulation_codes.html)
  * [ASE](https://atomistics.readthedocs.io/en/latest/simulation_codes.html#ase)
  * [LAMMPS](https://atomistics.readthedocs.io/en/latest/simulation_codes.html#lammps)
  * [Quantum Espresso](https://atomistics.readthedocs.io/en/latest/simulation_codes.html#id1)
* [Workflows](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html)
  * [Elastic Matrix](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#elastic-matrix)
  * [Energy Volume Curve](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#energy-volume-curve)
  * [Molecular Dynamics](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#molecular-dynamics)
  * [Harmonic Approximation](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#harmonic-approximation)
  * [Structure Optimization](https://atomistics.readthedocs.io/en/latest/lammps_workflows.html#structure-optimization)
* [Materials Properties](https://atomistics.readthedocs.io/en/latest/materialproperties.html)
  * [Elastic Properties](https://atomistics.readthedocs.io/en/latest/bulk_modulus_with_gpaw.html)
  * [Thermal Expansion](https://atomistics.readthedocs.io/en/latest/thermal_expansion_with_lammps.html)
  * [Helmholtz Free Energy](https://atomistics.readthedocs.io/en/latest/free_energy_calculation.html)
  * [Phase Diagram](https://atomistics.readthedocs.io/en/latest/phasediagram.html)
