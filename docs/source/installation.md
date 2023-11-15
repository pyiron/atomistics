# Installation 
The `atomistics` package is a pure python code, which can be installed either via the Python Package Index (pypi) or via
the `conda` package manager. Still the `atomistics` package only contains interfaces to the corresponding simulation 
codes, not the simulation codes itself. Consequently, an installation via the `conda` package manager is recommended as
it allows to install many opensource simulation codes in the same environment as the `atomistics` package without the 
need to compile the simulation code manually. The `conda` packages for these simulation codes are maintained by the
developers of the `atomistics` package in collaboration with the developers of the corresponding simulation codes. 

## conda-based Installation 
For the conda-based installation both the `atomistics` package as well as the corresponding simulation codes are 
distributed via the [conda-forge](https://conda-forge.org) community channel. By specifying the option `-c conda-forge`
the `conda` package manager installs the dependencies from the [conda-forge](https://conda-forge.org) community channel.
```
conda install -c conda-forge atomistics
```
As the `atomistics` package depends on the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) the effective
medium theory potential simulation code [EMT](https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html) is automatically 
installed with the basic installation of the `atomistics` package. In the following the simulation codes are sorted in 
alphabetical order: 

### Abinit 
[Abinit](https://www.abinit.org) - Plane wave density functional theory:
```
conda install -c conda-forge abinit
```

### GPAW 
[GPAW](https://wiki.fysik.dtu.dk/gpaw/) - Density functional theory Python code based on the projector-augmented wave 
method:
```
conda install -c conda-forge gpaw
```

### LAMMPS 
[LAMMPS](https://www.lammps.org) - Molecular Dynamics:
```
conda install -c conda-forge lammps pylammpsmpi jinja2 pandas iprpy-data
```
In addition to the conda package for the [LAMMPS](https://www.lammps.org) simulation code the interface in the 
`atomistics` package uses the [pylammpsmpi](https://github.com/pyiron/pylammpsmpi), [jinja2](https://jinja.palletsprojects.com/) 
templates, the [pandas DataFrames](https://pandas.pydata.org) to represent interatomic potentials, and finally it can 
leverage the [NIST database for interatomic potentials](https://www.ctcms.nist.gov/potentials) via the `iprpy-data`
package, which includes a wide range of interatomic potentials. The `iprpy-data` package is optional. 

### Quantum Espresso 
[Quantum Espresso](https://www.quantum-espresso.org) - Integrated suite of Open-Source computer codes for 
electronic-structure calculations:
```
conda install -c conda-forge qe
```

### Siesta
[Siesta](https://siesta-project.org) - Electronic structure calculations and ab initio molecular dynamics:
```
conda install -c conda-forge siesta
```

### Phonopy
[Phonopy](https://phonopy.github.io/phonopy/) - open source package for phonon calculations at harmonic and 
quasi-harmonic levels:
```
conda install -c phonopy seekpath structuretoolkit
```

## pypi-based Installation 
While the conda-based installation is recommended, it is also possible to install the `atomistics` package via the 
Python Package Index: 
```
pip install atomistics
```
As the `atomistics` package depends on the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) the effective
medium theory potential simulation code [EMT](https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html) is automatically 
installed with the basic installation of the `atomistics` package. 

Beyond the basic installation of the `atomistics` package it is also possible to install the extra requirements for 
specific simulation codes directly from the Python Package Index. It is important to mention that apart from the 
[GPAW](https://wiki.fysik.dtu.dk/gpaw/) simulation code, the simulation codes are not distributed via the Python Package
Index, so they have to be installed separately. 
### GPAW
[GPAW](https://wiki.fysik.dtu.dk/gpaw/) - Density functional theory Python code based on the projector-augmented wave 
method
```
pip install atomistics[gpaw]
```

### LAMMPS
[LAMMPS](https://www.lammps.org) - Molecular Dynamics:
```
pip install atomistics[lammps]
```

### Phonopy
[Phonopy](https://phonopy.github.io/phonopy/) - open source package for phonon calculations at harmonic and 
quasi-harmonic levels:
```
pip install atomistics[phonopy]
```