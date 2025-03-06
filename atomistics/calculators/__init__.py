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

__all__ = [
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


try:
    from atomistics.calculators.qe import (
        calc_static_with_qe,
        evaluate_with_qe,
        optimize_positions_and_volume_with_qe,
    )

    __all__ += [
        "calc_static_with_qe",
        "evaluate_with_qe",
        "optimize_positions_and_volume_with_qe",
    ]
except ImportError:
    pass

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

    __all__ += [
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
except ImportError:
    pass

try:
    from atomistics.calculators.lammps.phonon import (
        calc_molecular_dynamics_phonons_with_lammpslib,
    )

    __all__ += ["calc_molecular_dynamics_phonons_with_lammpslib"]
except ImportError:
    pass

try:
    from atomistics.calculators.sphinxdft import (
        evaluate_with_sphinx,
    )

    __all__ += [
        "evaluate_with_sphinx",
    ]
except ImportError:
    pass

try:
    from atomistics.calculators.vasp import (
        calc_static_with_vasp,
        evaluate_with_vasp,
        optimize_cell_with_vasp,
        optimize_positions_and_volume_with_vasp,
        optimize_positions_with_vasp,
        optimize_volume_with_vasp,
    )

    __all__ += [
        "evaluate_with_vasp",
        "calc_static_with_vasp",
        "optimize_positions_and_volume_with_vasp",
        "optimize_positions_with_vasp",
        "optimize_cell_with_vasp",
        "optimize_volume_with_vasp",
    ]
except ImportError:
    pass
