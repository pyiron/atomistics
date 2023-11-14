# Materials Properties
Demonstrate the calculation of common material properties with the `atomistics` package. The examples use different 
simulation codes, still the examples are not simulation code specific. It is one of the core features of the `atomistics`
package that all simulation workflow to calculate a specific material property can be executed with all simulation codes.

## Elastic Properties
Calculate the elastic properties for Aluminium using the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) DFT code.

### Equation of State 
Calculate the change of potential energy in dependence of the unit cell volume to identify the minimum as the equilibrium
volume. The energy at the equilibrium volume gives the equilibrium energy and the derivative gives the bulk modulus. 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import EnergyVolumeCurveWorkflow
from gpaw import GPAW, PW
import numpy as np


calculator = EnergyVolumeCurveWorkflow(
    structure=bulk("Al", a=4.05, cubic=True),
    num_points=11,
    fit_type='polynomial',
    fit_order=3,
    vol_range=0.05,
    axes=['x', 'y', 'z'],
    strains=None,
)
task_dict = calculator.generate_structures()
result_dict = evaluate_with_ase(
    task_dict=task_dict,
    ase_calculator=GPAW(
        xc="PBE",
        mode=PW(300),
        kpts=(3, 3, 3)
    )
)
fit_dict = calculator.analyse_structures(output_dict=result_dict)
print(fit_dict)
```

### Elastic Matrix
An alternative approach to calculate the bulk modulus is based on the relation `B = (1/3) (C11 + 2 C12 )`. The bulk
modulus can be calculated based on the sum of the first elastic constant `C11` and twice the second elastic constant `C12`
divided by there. 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import ElasticMatrixWorkflow
from gpaw import GPAW, PW
import numpy as np


calculator = ElasticMatrixWorkflow(
    structure=bulk("Al", a=4.0, cubic=True),
    num_of_point=5,
    eps_range=0.05,
    sqrt_eta=True,
    fit_order=2
)
task_dict = calculator.generate_structures()
result_dict = evaluate_with_ase(
    task_dict=task_dict,
    ase_calculator=GPAW(
        xc="PBE",
        mode=PW(300),
        kpts=(3, 3, 3)
    )
)
elastic_dict = calculator.analyse_structures(output_dict=result_dict)
print(elastic_dict)
```

## Thermal Expansion 
Calculate the thermal expansion for a Morse Pair potential using the [LAMMPS](https://www.lammps.org/) molecular dynamics
simulation code. 

Import required software packages: 
```
from ase.build import bulk
from atomistics.calculators.lammps import evaluate_with_lammps, get_potential_dataframe
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow
from atomistics.workflows.quasiharmonic.workflow import QuasiHarmonicWorkflow
from atomistics.shared.thermo.debye import get_debye_model
from atomistics.shared.thermo.thermo import get_thermo_bulk_model
import numpy as np
import pandas
from phonopy.units import VaspToTHz
```

Define the Morse Potential:
```
element = "Al"
alpha = 1.8
r0 = 2.95
D0 = 0.5
cutoff = 9.0
potential_dataframe = pandas.DataFrame({
    "Config": [[
        "pair_style morse/smooth/linear %f"%cutoff,
        "pair_coeff * * %.16f %.16f %.16f"%(D0, alpha, r0)
    ]],
    "Filename": [[]],
    "Model": ["Morse"],
    "Name": ["Morse"],
    "Species": [["Al"]],
}).iloc[0]
potential_dataframe
```
### Equation of State 
```
structure = bulk("Al", a=4.05, cubic=True)
workflow = EnergyVolumeCurveWorkflow(
    structure=structure,
    num_points=11,
    fit_type='polynomial',
    fit_order=3,
    vol_range=0.05,
    axes=['x', 'y', 'z'],
    strains=None,
)
structure_dict = workflow.generate_structures()
result_dict = evaluate_with_lammps(
    task_dict=structure_dict, 
    potential_dataframe=potential_dataframe
)
fit_dict = workflow.analyse_structures(output_dict=result_dict)
debye_model = get_debye_model(fit_dict=fit_dict, masses=structure.get_masses(), num_steps=50)
T_debye_low, T_debye_high = debye_model.debye_temperature
pes = get_thermo_bulk_model(
    temperatures=np.linspace(1, 1500, 200),
    debye_model=debye_model,
)
pes.plot_contourf(show_min_erg_path=True)
```
### Quasi-Harmonic Approximation 
```
calculator = QuasiHarmonicWorkflow(
    structure=structure,
    num_points=11,
    vol_range=0.05,
    interaction_range=10,
    factor=VaspToTHz,
    displacement=0.01,
    dos_mesh=20,
    primitive_matrix=None,
    number_of_snapshots=None,
)
structure_dict = calculator.generate_structures()
result_dict = evaluate_with_lammps(
    task_dict=structure_dict,
    potential_dataframe=potential_dataframe,
)
eng_internal_dict, mesh_collect_dict, dos_collect_dict = calculator.analyse_structures(output_dict=result_dict)
tp_collect_dict = calculator.get_thermal_properties(t_min=1, t_max=1500, t_step=50, temperatures=None)  

temperatures = tp_collect_dict[1.0]['temperatures']
temperature_max = max(temperatures)
strain_lst = eng_internal_dict.keys()
volume_lst = calculator.get_volume_lst()
eng_int_lst = np.array(list(eng_internal_dict.values()))

fit_deg = 4
vol_best = volume_lst[int(len(volume_lst)/2)]
vol_lst, eng_lst = [], []
for i, temp in enumerate(temperatures):
    free_eng_lst = np.array([tp_collect_dict[s]['free_energy'][i] for s in strain_lst]) + eng_int_lst
    p = np.polyfit(volume_lst, free_eng_lst, deg=fit_deg)
    extrema = np.roots(np.polyder(p, m=1)).real
    vol_select = extrema[np.argmin(np.abs(extrema - vol_best))]
    eng_lst.append(np.poly1d(p)(vol_select))
    vol_lst.append(vol_select)
    
fig, ax=plt.subplots(1,1)
for i, temp in enumerate(temperatures):
    ax.plot(volume_lst, np.array([
        tp_collect_dict[s]['free_energy'][i] 
        for s in strain_lst
    ]) + eng_int_lst, color=cmap(temp/temperature_max))
ax.set_xlabel("Volume")
ax.set_ylabel("Free energy  ($U + F_{vib}$) [eV]")
normalize = matplotlib.colors.Normalize(vmin=np.min(temperatures), vmax=np.max(temperatures))
scalarmappaple = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
scalarmappaple.set_array(list(tp_collect_dict.keys()))
cbar = plt.colorbar(scalarmappaple, ax=ax)
cbar.set_label("Temperature")
plt.plot(vol_lst, eng_lst, color="black", linestyle="--")
```
### Molecular Dynamics
For the pair potential we find good agreement between the three different approximation

## Phase Diagram 
One of the goals of the `atomistics` package is to be able to calculate phase diagrams with ab-initio precision. 

### Quasi-Harmonic Approximation 
coming soon

### Calphy 
coming soon 