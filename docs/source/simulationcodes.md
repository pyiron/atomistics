# Simulation Codes

## ASE
At the current stage the majority of simulation codes are interfaced using the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).
The limitation of the ASE based interfaces is that the simulation codes are only used to calculate energies, forces and
stresses, while more complex computations like structure optimization or molecular dynamics are implemented in python.

### Abinit
[Abinit](https://www.abinit.org) - Plane wave density functional theory:
```
from ase.calculators.abinit import Abinit
from ase.units import Ry
from atomistics.calculators import evaluate_with_ase

result_dict = evaluate_with_ase(
    task_dict={},
    ase_calculator=Abinit(
        label='abinit_evcurve',
        nbands=32,
        ecut=10 * Ry,
        kpts=(3, 3, 3),
        toldfe=1.0e-2,
        v8_legacy_format=False,
    )
)
```
The full documentation of the corresponding interface is available on the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/ase/calculators/abinit.html)
website. 

### EMT
[EMT](https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html) - Effective medium theory: 
```
from ase.calculators.emt import EMT
from atomistics.calculators import evaluate_with_ase

result_dict = evaluate_with_ase(
    task_dict={}, 
    ase_calculator=EMT()
)
```
The full documentation of the corresponding interface is available on the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html)
website. 

### GPAW
[GPAW](https://wiki.fysik.dtu.dk/gpaw/) - Density functional theory Python code based on the projector-augmented wave 
method:
```
from gpaw import GPAW, PW
from atomistics.calculators import evaluate_with_ase

result_dict = evaluate_with_ase(
    task_dict={}, 
    ase_calculator=GPAW(
        xc="PBE",
        mode=PW(300),
        kpts=(3, 3, 3)
    )
)
```
The full documentation of the corresponding interface is available on the [GPAW](https://wiki.fysik.dtu.dk/gpaw/)
website.

### Quantum Espresso 
[Quantum Espresso](https://www.quantum-espresso.org) - Integrated suite of Open-Source computer codes for 
electronic-structure calculations:
```
from ase.calculators.espresso import Espresso
from atomistics.calculators import evaluate_with_ase

result_dict = evaluate_with_ase(
    task_dict={}, 
    ase_calculator=Espresso(
        pseudopotentials={"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
        tstress=True,
        tprnfor=True,
        kpts=(3, 3, 3),
    )
)
```
The full documentation of the corresponding interface is available on the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html)
website. 

### Siesta
[Siesta](https://siesta-project.org) - Electronic structure calculations and ab initio molecular dynamics:
```
from ase.calculators.siesta import Siesta
from ase.units import Ry
from atomistics.calculators import evaluate_with_ase

result_dict = evaluate_with_ase(
    task_dict={}, 
    ase_calculator=Siesta(
        label="siesta",
        xc="PBE",
        mesh_cutoff=200 * Ry,
        energy_shift=0.01 * Ry,
        basis_set="DZ",
        kpts=(5, 5, 5),
        fdf_arguments={"DM.MixingWeight": 0.1, "MaxSCFIterations": 100},
        pseudo_path=os.path.abspath("tests/static/siesta"),
        pseudo_qualifier="",
    )
)
```
The full documentation of the corresponding interface is available on the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/ase/calculators/siesta.html)
website.

## LAMMPS
[LAMMPS](https://www.lammps.org) - Molecular Dynamics:
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_lammps, get_potential_by_name

structure = bulk("Al", cubic=True)
potential_dataframe = get_potential_by_name(
    potential_name='1999--Mishin-Y--Al--LAMMPS--ipr1'
)

result_dict = evaluate_with_lammps(
    task_dict={},
    potential_dataframe=potential_dataframe,
)
```
The [LAMMPS](https://www.lammps.org) interface is based on the [pylammpsmpi](https://github.com/pyiron/pylammpsmpi)
package which couples a [LAMMPS](https://www.lammps.org) instance which is parallelized via the Message Passing Interface
(MPI) with a serial python process or jupyter notebook. The challenging part about molecular dynamics simulation is 
identifying a suitable interatomic potential. 

To address this challenge the `atomistics` package is leveraging the [NIST database of interatomic potentials](https://www.ctcms.nist.gov/potentials). 
It is recommended to install this database `iprpy-data` via the `conda` package manager, then the `resource_path` is
automatically set to `${CONDA_PREFIX}/share/iprpy`. Alternatively, the `resource_path` can be specified manually as an
optional parameter of the `get_potential_by_name()` function.

In addition, the `get_potential_dataframe(structure)` function which takes an `ase.atoms.Atoms` object as input can be
used to query the [NIST database of interatomic potentials](https://www.ctcms.nist.gov/potentials) for potentials, which
include the interatomic interactions required to simulate the atomic structure defined by the `ase.atoms.Atoms` object. 
It returns a `pandas.DataFrame` with all the available potentials and the `resource_path` can again be specified as 
optional parameter.

Finally, another option to specify the interatomic potential for a LAMMPS simulation is by defining the `potential_dataframe`
directly: 
```
potential_dataframe = pandas.DataFrame({
    "Config": [[
        "pair_style morse/smooth/linear 9.0",
        "pair_coeff * * 0.5 1.8 2.95"
    ]],
    "Filename": [[]],
    "Model": ["Morse"],
    "Name": ["Morse"],
    "Species": [["Al"]],
})
```

## Quantum Espresso
[Quantum Espresso](https://www.quantum-espresso.org) - Integrated suite of Open-Source computer codes for 
electronic-structure calculations:
```
from atomistics.calculators import evaluate_with_qe

result_dict = evaluate_with_qe(
    task_dict={},
    calculation_name="espresso",
    working_directory=".",
    kpts=(3, 3, 3),
    pseudopotentials={
        "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"
    },
    tstress=True,
    tprnfor=True,
    ecutwfc=40.0,          # kinetic energy cutoff (Ry) for wavefunctions
    conv_thr=1e-06,        # Convergence threshold for selfconsistency
    diagonalization='david',  
    electron_maxstep=100,  # maximum number of iterations in a scf step. 
    nstep=200,             # number of molecular-dynamics or structural optimization steps performed in this run.
    etot_conv_thr=1e-4,    # Convergence threshold on total energy (a.u) for ionic minimization
    forc_conv_thr=1e-3,    # Convergence threshold on forces (a.u) for ionic minimization
    smearing='gaussian',   # ordinary Gaussian spreading (Default)
)
```
This secondary interface for [Quantum Espresso](https://www.quantum-espresso.org) is based on the input writer from the
[Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) and the output is parsed using the [pwtools](https://elcorto.github.io/pwtools/).
The executable can be set using the `ASE_ESPRESSO_COMMAND` environment variable:
```
export ASE_ESPRESSO_COMMAND="pw.x -in PREFIX.pwi > PREFIX.pwo"
```
The full list of possible keyword arguments is available in the [Quantum Espresso Documentation](https://www.quantum-espresso.org/Doc/INPUT_PW.html).
Finally, the [Standard solid-state pseudopotentials (SSSP)](https://www.materialscloud.org/discover/sssp/table/efficiency) 
for quantum espresso are distributed via the materials cloud.