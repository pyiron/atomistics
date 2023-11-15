# Materials Properties
Demonstrate the calculation of common material properties with the `atomistics` package. The examples use different 
simulation codes, still the examples are not simulation code specific. It is one of the core features of the `atomistics`
package that all simulation workflow to calculate a specific material property can be executed with all simulation codes.

## Elastic Properties
Calculate the bulk modulus for Aluminium using the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) DFT code:

### Equation of State 
One way to calculate the bulk modulus is using the Equation of State to calculate the equilibrium properties:
```
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
be executed by the calculator. In this case the task is to calculate the energy `calc_energy` of the eleven generated 
structures. Each structure is labeled by the ratio of compression or elongation. In the second step the `task_dict` 
is evaluated with the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) simulation code using the `evaluate_with_ase()` function:
```
result_dict = evaluate_with_ase(
    task_dict=task_dict,
    ase_calculator=GPAW(
        xc="PBE",
        mode=PW(300),
        kpts=(3, 3, 3)
    )
)
print(result_dict)
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
In analogy to the `task_dict` which defines the tasks to be executed by the simulation code the `result_dict` summarizes 
the results of the calculations. In this case the energies calculated for the specific strains. By ordering both the 
`task_dict` and the `result_dict` with the same labels, the `EnergyVolumeCurveWorkflow` object is able to match the 
calculation results to the corresponding structure. Finally, in the third step the `analyse_structures()` function takes
the `result_dict` as an input and fits the Equation of State with the fitting parameters defined in the first step:
```
fit_dict = workflow.analyse_structures(output_dict=result_dict)
print(fit_dict)
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
The bulk modulus for Aluminium is calculated using the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) simulation code by fitting
the Equation of State with a third order polynomial over a volume range of +/-5% to be 72.3GPa.  

### Elastic Matrix
An alternative approach to calculate the bulk modulus is based on the relation `B = (1/3) (C11 + 2 C12 )`. The bulk
modulus can be calculated based on the sum of the first elastic constant `C11` and twice the second elastic constant `C12`
divided by there. 
```
from ase.build import bulk
from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import ElasticMatrixWorkflow
from gpaw import GPAW, PW

workflow = ElasticMatrixWorkflow(
    structure=bulk("Al", a=4.05, cubic=True),
    num_of_point=5,
    eps_range=0.05,
    sqrt_eta=True,
    fit_order=2
)
task_dict = workflow.generate_structures()
print(task_dict)
>>> {'calc_energy': OrderedDict([
>>>     ('s_e_0', Atoms(symbols='Al4', pbc=True, cell=[4.05, 4.05, 4.05])), 
>>>     ('s_01_e_m0_05000', Atoms(symbols='Al4', pbc=True, cell=[3.8421673571095107, 3.8421673571095107, 3.8421673571095107])), 
>>>     ('s_01_e_m0_02500', Atoms(symbols='Al4', pbc=True, cell=[3.94745170964797, 3.94745170964797, 3.94745170964797])), 
>>>     ('s_01_e_0_02500', Atoms(symbols='Al4', pbc=True, cell=[4.150015060213919, 4.150015060213919, 4.150015060213919])), 
>>>     ('s_01_e_0_05000', Atoms(symbols='Al4', pbc=True, cell=[4.247675835085893, 4.247675835085893, 4.247675835085893])), 
>>>     ('s_08_e_m0_05000', Atoms(symbols='Al4', pbc=True, cell=[3.8421673571095107, 3.8421673571095107, 4.05])), 
>>>     ('s_08_e_m0_02500', Atoms(symbols='Al4', pbc=True, cell=[3.94745170964797, 3.94745170964797, 4.05])), 
>>>     ('s_08_e_0_02500', Atoms(symbols='Al4', pbc=True, cell=[4.150015060213919, 4.150015060213919, 4.05])), 
>>>     ('s_08_e_0_05000', Atoms(symbols='Al4', pbc=True, cell=[4.247675835085893, 4.247675835085893, 4.05])), 
>>>     ('s_23_e_m0_05000', Atoms(symbols='Al4', pbc=True, cell=[
>>>         [4.039260597921188, -0.2084152371679185, -0.2084152371679185], 
>>>         [-0.2084152371679185, 4.039260597921188, -0.2084152371679185], 
>>>         [-0.2084152371679185, -0.2084152371679185, 4.039260597921188]
>>>     ])), 
>>>     ('s_23_e_m0_02500', Atoms(symbols='Al4', pbc=True, cell=[
>>>         [4.047399159178924, -0.1026159010347065, -0.1026159010347065], 
>>>         [-0.1026159010347065, 4.047399159178924, -0.1026159010347065], 
>>>         [-0.1026159010347065, -0.1026159010347065, 4.047399159178924]
>>>     ])), 
>>>     ('s_23_e_0_02500', Atoms(symbols='Al4', pbc=True, cell=[
>>>         [4.047526418127057, 0.1000747084794181, 0.1000747084794181], 
>>>         [0.1000747084794181, 4.047526418127057, 0.1000747084794181], 
>>>         [0.1000747084794181, 0.1000747084794181, 4.047526418127057]
>>>     ])), 
>>>     ('s_23_e_0_05000', Atoms(symbols='Al4', pbc=True, cell=[
>>>         [4.0402958099962145, 0.19812845289162093, 0.19812845289162093], 
>>>         [0.19812845289162093, 4.0402958099962145, 0.19812845289162093], 
>>>         [0.19812845289162093, 0.19812845289162093, 4.0402958099962145]
>>>     ]))
>>> ])}
```
In analogy to the example with the `EnergyVolumeCurveWorkflow` above, the `ElasticMatrixWorkflow` is initialized with all
the parameters required to generate the atomistic structures and afterwards fit the resulting energies. By calling the
`generate_structures()` function the task dictionary `task_dict` is generated. The task dictionary specifies that the 
energy should be calculated for a total of thirteen structures with different displacements. In the second step the 
structures are again evaluated with the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) simulation code: 
```
result_dict = evaluate_with_ase(
    task_dict=task_dict,
    ase_calculator=GPAW(
        xc="PBE",
        mode=PW(300),
        kpts=(3, 3, 3)
    )
)
print(result_dict)
>>> {'energy': {
>>>     's_e_0': -14.936666396364958, 
>>>     's_01_e_m0_05000': -14.509157650668122, 
>>>     's_01_e_m0_02500': -14.841982287144095, 
>>>     's_01_e_0_02500': -14.861851384196036, 
>>>     's_01_e_0_05000': -14.667794842771894, 
>>>     's_08_e_m0_05000': -14.761984597147846, 
>>>     's_08_e_m0_02500': -14.915410385310373, 
>>>     's_08_e_0_02500': -14.906256779097374, 
>>>     's_08_e_0_05000': -14.792358225782438, 
>>>     's_23_e_m0_05000': -14.276020694686991, 
>>>     's_23_e_m0_02500': -14.82856618064028, 
>>>     's_23_e_0_02500': -14.919070452898067, 
>>>     's_23_e_0_05000': -14.61301941504334
>>> }}
```
The atomistic structures are evaluated with the `evaluate_with_ase()` function, which returns the `result_dict`. This 
`result_dict` in analogy to the `task_dict` contains the same keys as well as the energies calculated with the 
[GPAW](https://wiki.fysik.dtu.dk/gpaw/) simulation code. Finally, the `result_dict` is provided as an input to the 
`analyse_structures()` function to calculate the corresponding elastic constants: 
```
elastic_dict = workflow.analyse_structures(output_dict=result_dict)
print(elastic_dict)
>>> OrderedDict([
>>>     ('SGN', 225), 
>>>     ('v0', 66.43012500000002), 
>>>     ('LC', 'CI'), 
>>>     ('Lag_strain_list', ['01', '08', '23']), 
>>>     ('epss', array([-0.05 , -0.025,  0.   ,  0.025,  0.05 ])), 
>>>     ('e0', -14.936666396364958), 
>>>     ('strain_energy', [
>>>         [
>>>             (-0.05, -14.509157650668122), 
>>>             (-0.025, -14.841982287144095), 
>>>             (0.0, -14.936666396364958), 
>>>             (0.02500000000000001, -14.861851384196036), 
>>>             (0.05, -14.667794842771894)
>>>         ], 
>>>         [
>>>             (-0.05, -14.761984597147846), 
>>>             (-0.025, -14.915410385310373), 
>>>             (0.0, -14.936666396364958), 
>>>             (0.02500000000000001, -14.906256779097374), 
>>>             (0.05, -14.792358225782438)
>>>         ], 
>>>         [
>>>             (-0.05, -14.276020694686991), 
>>>             (-0.025, -14.82856618064028), 
>>>             (0.0, -14.936666396364958), 
>>>             (0.02500000000000001, -14.919070452898067), 
>>>             (0.05, -14.61301941504334)
>>>         ]
>>>     ]), 
>>>     ('C', array([
>>>         [98.43569795, 63.17412931, 63.17412931,  0.        ,  0.        ,  0.        ],
>>>         [63.17412931, 98.43569795, 63.17412931,  0.        ,  0.        ,  0.        ],
>>>         [63.17412931, 63.17412931, 98.43569795,  0.        ,  0.        ,  0.        ],
>>>         [ 0.        ,  0.        ,  0.        , 84.66136128,  0.        ,  0.        ],
>>>         [ 0.        ,  0.        ,  0.        ,  0.        , 84.66136128,  0.        ],
>>>         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 84.66136128]
>>>     ])), 
>>>     ('A2', array([2.10448666, 1.0086892 , 3.17048793])), 
>>>     ('BV', 74.92798552228444), 
>>>     ('GV', 57.8491304939761), 
>>>     ('EV', 138.02584019743034), 
>>>     ('nuV', 0.19298111327535986), 
>>>     ('S', array([
>>>         [ 0.02038923, -0.00797026, -0.00797026,  0.        ,  0.        ,  0.        ],
>>>         [-0.00797026,  0.02038923, -0.00797026,  0.        ,  0.        ,  0.        ],
>>>         [-0.00797026, -0.00797026,  0.02038923,  0.        ,  0.        ,  0.        ],
>>>         [ 0.        ,  0.        ,  0.        ,  0.01181176,  0.        ,  0.        ],
>>>         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.01181176,  0.        ],
>>>         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.01181176]
>>>     ])), 
>>>     ('BR', 74.9279855222844), 
>>>     ('GR', 33.5856196420454), 
>>>     ('ER', 87.65941305083547), 
>>>     ('nuR', 0.30501408020913495), 
>>>     ('BH', 74.92798552228442), 
>>>     ('GH', 45.71737506801075), 
>>>     ('EH', 113.97207240565497), 
>>>     ('nuH', 0.246485304942663), 
>>>     ('AVR', 26.536421673199147), 
>>>     ('C_eigval', EigResult(
>>>         eigenvalues=array([ 35.26156864, 224.78395657,  35.26156864,  84.66136128,  84.66136128,  84.66136128]), 
>>>         eigenvectors=array([
>>>             [-0.81649658,  0.57735027, -0.15564171,  0.        ,  0.        ,  0.        ],
>>>             [ 0.40824829,  0.57735027, -0.61632016,  0.        ,  0.        ,  0.        ],
>>>             [ 0.40824829,  0.57735027,  0.77196187,  0.        ,  0.        ,  0.        ],
>>>             [ 0.        ,  0.        ,  0.        ,  1.        ,  0.        ,  0.        ],
>>>             [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
>>>             [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  1.        ]
>>>         ])
>>>     ))
>>> ])
```
The bulk modulus calculated from the elastic constants `C11` and `C12` based on a strain of +/- 5% is calculated with 
the [GPAW](https://wiki.fysik.dtu.dk/gpaw/) simulation code to be 74.9GPa. This differs from the bulk modulus calculated
from the Equation of State above by 2.6GPa. In comparison to the experimental bulk modulus for Aluminium which is
[reported to be 76GPa](https://periodictable.com/Elements/013/data.html) the calculation based on the elastic constants
seem to be more precise, still this is more likely related to error cancellation. In general elastic properties calculated
from density functional theory are expected to have errors of about 5-10% unless carefully converged.

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
workflow = QuasiHarmonicWorkflow(
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
structure_dict = workflow.generate_structures()
result_dict = evaluate_with_lammps(
    task_dict=structure_dict,
    potential_dataframe=potential_dataframe,
)
eng_internal_dict, mesh_collect_dict, dos_collect_dict = workflow.analyse_structures(output_dict=result_dict)
tp_collect_dict = workflow.get_thermal_properties(t_min=1, t_max=1500, t_step=50, temperatures=None)  

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