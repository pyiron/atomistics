from atomistics.calculators.ase import (
    calc_molecular_dynamics_langevin_with_ase,
    calc_molecular_dynamics_npt_with_ase,
    calc_molecular_dynamics_thermal_expansion_with_ase,
    calc_static_with_ase,
    evaluate_with_ase,
    optimize_positions_and_volume_with_ase,
    optimize_positions_with_ase,
    optimize_volume_with_ase,
)
from atomistics.calculators.hessian import evaluate_with_hessian
from atomistics.shared.import_warning import raise_warning

__all__: list[str] = [
    "calc_molecular_dynamics_langevin_with_ase",
    "calc_molecular_dynamics_npt_with_ase",
    "calc_molecular_dynamics_thermal_expansion_with_ase",
    "calc_static_with_ase",
    "evaluate_with_ase",
    "optimize_positions_with_ase",
    "optimize_positions_and_volume_with_ase",
    "optimize_volume_with_ase",
    "evaluate_with_hessian",
]
lammps_functions: list[str] = [
    "calc_molecular_dynamics_thermal_expansion_with_lammpslib",
    "calc_molecular_dynamics_nph_with_lammpslib",
    "calc_molecular_dynamics_npt_with_lammpslib",
    "calc_molecular_dynamics_nvt_with_lammpslib",
    "calc_molecular_dynamics_langevin_with_lammpslib",
    "calc_static_with_lammpslib",
    "evaluate_with_lammpslib",
    "evaluate_with_lammpslib_library_interface",
    "get_potential_dataframe",
    "get_potential_by_name",
    "optimize_positions_and_volume_with_lammpslib",
    "optimize_positions_with_lammpslib",
]
lammps_phonon_functions: list[str] = ["calc_molecular_dynamics_phonons_with_lammpslib"]
quantum_espresso_function: list[str] = [
    "calc_static_with_qe",
    "evaluate_with_qe",
    "optimize_positions_and_volume_with_qe",
]
sphinx_functions: list[str] = ["evaluate_with_sphinx"]
vasp_functions: list[str] = [
    "evaluate_with_vasp",
    "calc_static_with_vasp",
    "optimize_positions_and_volume_with_vasp",
    "optimize_positions_with_vasp",
    "optimize_cell_with_vasp",
    "optimize_volume_with_vasp",
]


try:
    from atomistics.calculators.qe import (
        calc_static_with_qe,
        evaluate_with_qe,
        optimize_positions_and_volume_with_qe,
    )
except ImportError as e:
    raise_warning(module_list=quantum_espresso_function, import_error=e)
else:
    __all__ += quantum_espresso_function


try:
    from atomistics.calculators.lammps import (
        calc_molecular_dynamics_langevin_with_lammpslib,
        calc_molecular_dynamics_nph_with_lammpslib,
        calc_molecular_dynamics_npt_with_lammpslib,
        calc_molecular_dynamics_nvt_with_lammpslib,
        calc_molecular_dynamics_thermal_expansion_with_lammpslib,
        calc_static_with_lammpslib,
        evaluate_with_lammpslib,
        evaluate_with_lammpslib_library_interface,
        get_potential_by_name,
        get_potential_dataframe,
        optimize_positions_and_volume_with_lammpslib,
        optimize_positions_with_lammpslib,
    )
except ImportError as e:
    raise_warning(module_list=lammps_functions, import_error=e)
else:
    __all__ += lammps_functions


try:
    from atomistics.calculators.lammps.phonon import (
        calc_molecular_dynamics_phonons_with_lammpslib,
    )
except ImportError as e:
    raise_warning(module_list=lammps_phonon_functions, import_error=e)
else:
    __all__ += lammps_phonon_functions


try:
    from atomistics.calculators.sphinxdft import (
        evaluate_with_sphinx,
    )
except ImportError as e:
    raise_warning(module_list=sphinx_functions, import_error=e)
else:
    __all__ += sphinx_functions


try:
    from atomistics.calculators.vasp import (
        calc_static_with_vasp,
        evaluate_with_vasp,
        optimize_cell_with_vasp,
        optimize_positions_and_volume_with_vasp,
        optimize_positions_with_vasp,
        optimize_volume_with_vasp,
    )
except ImportError as e:
    raise_warning(module_list=vasp_functions, import_error=e)
else:
    __all__ += vasp_functions
