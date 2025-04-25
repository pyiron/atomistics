from atomistics.calculators.lammps.libcalculator import (
    calc_molecular_dynamics_langevin_with_lammpslib,
    calc_molecular_dynamics_nph_with_lammpslib,
    calc_molecular_dynamics_npt_with_lammpslib,
    calc_molecular_dynamics_nvt_with_lammpslib,
    calc_molecular_dynamics_thermal_expansion_with_lammpslib,
    calc_static_with_lammpslib,
    evaluate_with_lammpslib,
    evaluate_with_lammpslib_library_interface,
    optimize_positions_and_volume_with_lammpslib,
    optimize_positions_with_lammpslib,
)
from atomistics.calculators.lammps.potential import (
    get_potential_by_name,
    get_potential_dataframe,
)
from atomistics.shared.import_warning import raise_warning

__all__: list[str] = [
    "calc_molecular_dynamics_thermal_expansion_with_lammpslib",
    "calc_molecular_dynamics_nph_with_lammpslib",
    "calc_molecular_dynamics_npt_with_lammpslib",
    "calc_molecular_dynamics_nvt_with_lammpslib",
    "calc_molecular_dynamics_langevin_with_lammpslib",
    "calc_static_with_lammpslib",
    "evaluate_with_lammpslib",
    "evaluate_with_lammpslib_library_interface",
    "optimize_positions_and_volume_with_lammpslib",
    "optimize_positions_with_lammpslib",
    "get_potential_dataframe",
    "get_potential_by_name",
]
lammps_phonon_functions: list[str] = ["calc_molecular_dynamics_phonons_with_lammpslib"]


try:
    from atomistics.calculators.lammps.phonon import (
        calc_molecular_dynamics_phonons_with_lammpslib,
    )
except ImportError as e:
    raise_warning(module_list=lammps_phonon_functions, import_error=e)
else:
    __all__ += lammps_phonon_functions
