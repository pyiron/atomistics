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
simulation code. In the following three methods to calculate the thermal expansion are introduced and compared for a 
Morse Pair Potential for Aluminium. 

As a first step the potential is defined for the [LAMMPS](https://www.lammps.org/) molecular dynamics simulation code 
by specifying the `pair_style` and `pair_coeff` commands for the [Morse Pair Potential](https://docs.lammps.org/pair_morse.html)
as well as the Aluminium bulk structure: 
```
from ase.build import bulk
import pandas

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

structure = bulk("Al", a=4.05, cubic=True)
```
The `pandas.DataFrame` based format to specify interatomic potentials is the same `pylammpsmpi` uses to interface with 
the [NIST database for interatomic potentials](https://www.ctcms.nist.gov/potentials). In comparison to just providing
the `pair_style` and `pair_coeff` commands, this extended format enables referencing specific files for the interatomic
potentials `"Filename": [[]],` as well as the atomic species `"Species": [["Al"]],` to enable consistency checks if the 
interatomic potential implements all the interactions to simulate a given atomic structure. 

Finally, the last step of the preparation before starting the actual calculation is optimizing the interatomic structure. 
While for the Morse potential used in this example this is not necessary, it is essential for extending this example to
other interactomic potentials. For the structure optimization the `optimize_positions_and_volume()` function is imported
and applied on the `ase.atoms.Atoms` bulk structure for Aluminium:
```
from atomistics.workflows import optimize_positions_and_volume

task_dict = optimize_positions_and_volume(structure=structure)
task_dict
>>> {'optimize_positions_and_volume': Atoms(symbols='Al4', pbc=True, cell=[4.05, 4.05, 4.05])}
```
It returns a `task_dict` with a single task, the optimization of the positions and the volume of the Aluminium structure.
This task is executed with the [LAMMPS](https://www.lammps.org/) molecular dynamics simulation code using the 
`evaluate_with_lammps()` function:
```
from atomistics.calculators import evaluate_with_lammps

result_dict = evaluate_with_lammps(
    task_dict=task_dict,
    potential_dataframe=potential_dataframe,
)
structure_opt = result_dict["structure_with_optimized_positions_and_volume"]
```
The `result_dict` just contains a single element, the `ase.atoms.Atoms` structure object with optimized positions and 
volume. After this step the preparation is completed and the three different approximations can be compared in the following.

### Equation of State 
The first approximation to calculate the thermal expansion is based on the Equation of State derived by [Moruzzi, V. L. et al.](https://link.aps.org/doi/10.1103/PhysRevB.37.790).
So in analogy to the previous example of calculating the elastic properties from the Equation of State, the `EnergyVolumeCurveWorkflow`
is initialized with the default parameters: 
```
from atomistics.workflows import EnergyVolumeCurveWorkflow

workflow_ev = EnergyVolumeCurveWorkflow(
    structure=structure_opt,
    num_points=11,
    fit_type='polynomial',
    fit_order=3,
    vol_range=0.05,
    axes=['x', 'y', 'z'],
    strains=None,
)
structure_dict = workflow_ev.generate_structures()
print(structure_dict)
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
>>>     (1.05,Atoms(symbols='Al4', pbc=True, cell=[4.1164052451001565, 4.1164052451001565, 4.1164052451001565]))
>>> ])}
```
After the initialization the `generate_structures()` function is called to generate the atomistic structures which are
then in the second step evaluated with the [LAMMPS](https://www.lammps.org/) molecular dynamics simulation code to derive
the equilibrium properties: 
```
result_dict = evaluate_with_lammps(
    task_dict=structure_dict, 
    potential_dataframe=potential_dataframe
)
print(result_dict):
>>> {'energy': {
>>>     0.95: -14.619170288727801, 
>>>     0.96: -14.664457483479836, 
>>>     0.97: -14.697945635153152, 
>>>     0.98: -14.720448033206749, 
>>>     0.99: -14.732723972540498, 
>>>     1.0: -14.73548275794779, 
>>>     1.01: -14.729389420395107, 
>>>     1.02: -14.715066161138207, 
>>>     1.03: -14.693095226824505, 
>>>     1.04: -14.664021603093682, 
>>>     1.05: -14.628355511604163
>>> }}
```
While in the previous example the fit of the energy volume curve was used directly, here the output of the fit, in
particular the derived equilibrium properties are the input for the Debye model as defined by [Moruzzi, V. L. et al.](https://link.aps.org/doi/10.1103/PhysRevB.37.790):
```
import numpy as np

temperatures_ev, volume_ev = workflow_ev.get_thermal_expansion(
    output_dict=result_dict, 
    temperatures=np.arange(1, 1500, 50),
)
```
The output of the Debye model provides the change of the temperature specific optimal volume `volume_ev` which can be 
plotted over the temperature `temperatures_ev` to determine the thermal expansion. 

### Quasi-Harmonic Approximation 
While the [Moruzzi, V. L. et al.](https://link.aps.org/doi/10.1103/PhysRevB.37.790) approach based on the Einstein crystal
is limited to a single frequency, the quasi-harmonic model includes the volume dependent free energy. Inside the 
`atomistics` package the harmonic and quasi-harmonic model are implemented based on an interface to the [Phonopy](https://phonopy.github.io/phonopy/)
framework. Still the user interface is still structured in the same three steps of (1) generating structures, (2) evaluating 
these structures and (3) fitting the corresponding model. Starting with the initialization of the `QuasiHarmonicWorkflow`
which combines the `PhonopyWorkflow` with the `EnergyVolumeCurveWorkflow`:
```
from atomistics.workflows import QuasiHarmonicWorkflow
from phonopy.units import VaspToTHz

workflow_qh = QuasiHarmonicWorkflow(
    structure=structure_opt,
    num_points=11,
    vol_range=0.05,
    interaction_range=10,
    factor=VaspToTHz,
    displacement=0.01,
    dos_mesh=20,
    primitive_matrix=None,
    number_of_snapshots=None,
)
structure_dict = workflow_qh.generate_structures()
print(structure_dict)
>>> {
>>>     'calc_energy': OrderedDict([
>>>          (0.95, Atoms(symbols='Al4', pbc=True, cell=[3.9813426685908118, 3.9813426685908118, 3.9813426685908118])),
>>>          (0.96, Atoms(symbols='Al4', pbc=True, cell=[3.9952635604153612, 3.9952635604153612, 3.9952635604153612])),
>>>          (0.97, Atoms(symbols='Al4', pbc=True, cell=[4.009088111958974, 4.009088111958974, 4.009088111958974])),
>>>          (0.98, Atoms(symbols='Al4', pbc=True, cell=[4.022817972936038, 4.022817972936038, 4.022817972936038])),
>>>          (0.99, Atoms(symbols='Al4', pbc=True, cell=[4.036454748321015, 4.036454748321015, 4.036454748321015])),
>>>          (1.0, Atoms(symbols='Al4', pbc=True, cell=[4.05, 4.05, 4.05])),
>>>          (1.01, Atoms(symbols='Al4', pbc=True, cell=[4.063455248345461, 4.063455248345461, 4.063455248345461])),
>>>          (1.02, Atoms(symbols='Al4', pbc=True, cell=[4.076821973718458, 4.076821973718458, 4.076821973718458])),
>>>          (1.03, Atoms(symbols='Al4', pbc=True, cell=[4.0901016179023415, 4.0901016179023415, 4.0901016179023415])),
>>>          (1.04, Atoms(symbols='Al4', pbc=True, cell=[4.1032955854717175, 4.1032955854717175, 4.1032955854717175])),
>>>          (1.05,Atoms(symbols='Al4', pbc=True, cell=[4.1164052451001565, 4.1164052451001565, 4.1164052451001565]))
>>>     ]),
>>>     'calc_forces': {
>>>          (0.95, 0): Atoms(symbols='Al108', pbc=True, cell=[11.944028005772434, 11.944028005772434, 11.944028005772434]),
>>>          (0.96, 0): Atoms(symbols='Al108', pbc=True, cell=[11.985790681246083, 11.985790681246083, 11.985790681246083]),
>>>          (0.97, 0): Atoms(symbols='Al108', pbc=True, cell=[12.027264335876922, 12.027264335876922, 12.027264335876922]),
>>>          (0.98, 0): Atoms(symbols='Al108', pbc=True, cell=[12.068453918808114, 12.068453918808114, 12.068453918808114]),
>>>          (0.99, 0): Atoms(symbols='Al108', pbc=True, cell=[12.109364244963045, 12.109364244963045, 12.109364244963045]),
>>>          (1.0, 0): Atoms(symbols='Al108', pbc=True, cell=[12.149999999999999, 12.149999999999999, 12.149999999999999]),
>>>          (1.01, 0): Atoms(symbols='Al108', pbc=True, cell=[12.190365745036383, 12.190365745036383, 12.190365745036383]),
>>>          (1.02, 0): Atoms(symbols='Al108', pbc=True, cell=[12.230465921155373, 12.230465921155373, 12.230465921155373]),
>>>          (1.03, 0): Atoms(symbols='Al108', pbc=True, cell=[12.270304853707025, 12.270304853707025, 12.270304853707025]),
>>>          (1.04, 0): Atoms(symbols='Al108', pbc=True, cell=[12.309886756415153, 12.309886756415153, 12.309886756415153]),
>>>          (1.05, 0): Atoms(symbols='Al108', pbc=True, cell=[12.349215735300469, 12.349215735300469, 12.349215735300469])
>>>     }
>>> }
```
In contrast to the previous workflows which only used the `calc_energy` function of the simulation codes the `PhonopyWorkflow`
and correspondingly also the `QuasiHarmonicWorkflow` require the calculation of the forces `calc_forces` in addition to
the calculation of the energy. Still the general steps of the workflow remain the same: 
```
result_dict = evaluate_with_lammps(
    task_dict=structure_dict,
    potential_dataframe=potential_dataframe,
)
```
The `structure_dict` is evaluated with the [LAMMPS](https://www.lammps.org/) molecular dynamics simulation code to 
calculate the corresponding energies and forces. The output is not plotted here as the forces for the 108 atom cells 
result in 3x108 outputs per cell. Still the structure of the `result_dict` again follows the labels of the `structure_dict`
as explained before. Finally, in the third step the individual free energy curves at the different temperatures are 
fitted to determine the equilibrium volume at the given temperature using the `get_thermal_expansion()` function: 
```
temperatures_qh, volume_qh = workflow_qh.get_thermal_expansion(
    output_dict=result_dict, 
    temperatures=np.arange(1, 1500, 50),
)
```
The optimal volume at the different `temperatures` is stored in the `volume_qh` in analogy to the previous section.

### Molecular Dynamics
Finally, the third and most commonly used method to determine the volume expansion is using a molecular dynamics 
calculation. While the `atomistics` package already includes a `LangevinWorkflow` at this point we use the [Nose-Hoover
thermostat implemented in LAMMPS](https://docs.lammps.org/fix_nh.html) directly via the LAMMPS calculator interface. 
```
from atomistics.calculators import calc_molecular_dynamics_thermal_expansion_with_lammps

structure_md = structure_opt.repeat(11)
temperature_md_lst, volume_md_lst = calc_molecular_dynamics_thermal_expansion_with_lammps(
    structure=structure,                       # atomistic structure
    potential_dataframe=potential_dataframe,   # interatomic potential defined as pandas.DataFrame 
    Tstart=15,                                 # temperature to for initial velocity distribution
    Tstop=1500,                                # final temperature
    Tstep=5,                                   # temperature step
    Tdamp=0.1,                                 # temperature damping of the thermostat 
    run=100,                                   # number of MD steps for each temperature
    thermo=100,                                # print out from the thermostat
    timestep=0.001,                            # time step for molecular dynamics 
    Pstart=0.0,                                # initial pressure
    Pstop=0.0,                                 # final pressure 
    Pdamp=1.0,                                 # barostat damping 
    seed=4928459,                              # random seed 
    dist="gaussian",                           # Gaussian velocity distribution 
)
```
The `calc_molecular_dynamics_thermal_expansion_with_lammps()` function defines a loop over a vector of temperatures in 
5K steps. For each step 100 molecular dynamics steps are executed before the temperature is again increased by 5K. For 
~280 steps with the Morse Pair Potential this takes approximately 5 minutes on a single core. These simulations can be 
further accelerated by adding the `cores` parameter. The increase in computational cost is on the one hand related to 
the large number of force and energy calls and on the other hand to the size of the atomistic structure, as these 
simulations are typically executed with >5000 atoms rather than the 4 or 108 atoms in the other approximations. The 
volume for the individual temperatures is stored in the `volume_md_lst` list. 

### Summary
To visually compare the thermal expansion predicted by the three different approximations, the [matplotlib](https://matplotlib.org)
is used to plot the temperature over the volume:
```
import matplotlib.pyplot as plt
plt.plot(np.array(volume_md_lst)/len(structure_md) * len(structure_opt), temperature_md_lst, label="Molecular Dynamics", color="C2")
plt.plot(volume_qh, temperatures_qh, label="Quasi-Harmonic", color="C0")
plt.plot(volume_ev, temperatures_ev, label="Moruzzi Model", color="C1")
plt.legend()
plt.xlabel("Volume ($\AA^3$)")
plt.ylabel("Temperature (K)")
```
The result is visualized in the following graph:

![Compare Thermal Expansion](../pictures/thermalexpansion.png)

While the both the [Moruzzi, V. L. et al.](https://link.aps.org/doi/10.1103/PhysRevB.37.790) and the quasi-harmonic 
approach over-predict the volume expansion with increasing temperature the overall result is in reasonable agreement for
the Morse Pair Potential. 

## Phase Diagram 
One of the goals of the `atomistics` package is to be able to calculate phase diagrams with ab-initio precision. Coming 
soon. 
