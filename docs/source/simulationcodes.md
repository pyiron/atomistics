# Simulation Codes
At the current stage the majority of simulation codes are interfaced using the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)

## Ab init
```
from ase.calculators.abinit import Abinit
from atomistics.calculators import evaluate_with_ase

result_dict = evaluate_with_ase(
    task_dict={}, 
    ase_calculator=EMT()
)
```

## EMT
```
from ase.calculators.emt import EMT
from atomistics.calculators import evaluate_with_ase

result_dict = evaluate_with_ase(
    task_dict={}, 
    ase_calculator=EMT()
)
```

## GPAW
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

## LAMMPS
```
from atomistics.calculators import evaluate_with_lammps, get_potential_dataframe

potential = '1999--Mishin-Y--Al--LAMMPS--ipr1'
resource_path = os.path.join(os.path.dirname(__file__), "static", "lammps")
structure = bulk("Al", cubic=True)
df_pot = get_potential_dataframe(
    structure=structure,
    resource_path=resource_path
)
df_pot_selected = df_pot[df_pot.Name == potential].iloc[0]

result_dict = evaluate_with_lammps(
    task_dict={},
    potential_dataframe=df_pot_selected,
)
```

## Quantum Espresso 
```
from ase.calculators.espresso import Espresso
from atomistics.calculators import evaluate_with_ase

pseudopotentials = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"}
result_dict = evaluate_with_ase(
    task_dict={}, 
    ase_calculator=Espresso(
        pseudopotentials=pseudopotentials,
        tstress=True,
        tprnfor=True,
        kpts=(3, 3, 3),
    )
)
```

## Siesta
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