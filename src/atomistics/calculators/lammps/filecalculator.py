import os
from collections.abc import Iterable
from typing import Any, Callable, Optional

import pandas
from ase.atoms import Atoms
from jinja2 import Template
from lammpsparser import parse_lammps_output_files as _parse_lammps_output_files
from lammpsparser import write_lammps_structure as _write_lammps_structure

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.lammps.commands import (
    LAMMPS_MINIMIZE,
    LAMMPS_RUN,
    LAMMPS_THERMO,
    LAMMPS_THERMO_STYLE,
)
from atomistics.calculators.lammps.shared import get_box_relax_command
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputStatic

DUMP_COMMANDS = [
    "dump 1 all custom 100 dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
    'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
]


class GenericOutput:
    """Accessor for parsed LAMMPS file-calculator output.

    Args:
        output_dict (dict[str, Any]): Parsed output dictionary returned by
            ``lammpsparser.parse_lammps_output_files``.
    """

    def __init__(self, output_dict: dict[str, Any]):
        self._output_dict = output_dict

    def get_forces(self) -> list[list[float]]:
        """Return atomic forces from the last recorded frame in eV/Å."""
        return self._output_dict["generic"]["forces"][-1]

    def get_energy_pot(self) -> float:
        """Return the potential energy from the last recorded frame in eV."""
        return self._output_dict["generic"]["energy_pot"][-1]

    def get_stress(self) -> list[float]:
        """Return the pressure tensor from the last recorded frame in GPa."""
        return self._output_dict["generic"]["pressures"][-1]

    def get_volume(self) -> float:
        """Return the cell volume from the last recorded frame in Å³."""
        return self._output_dict["generic"]["volume"][-1]


def _lammps_file_initialization(structure: Atoms) -> list[str]:
    """
    Generate the header LAMMPS input commands for a metal-units simulation.

    Sets up units, dimension, periodic boundary flags (derived from ``structure.pbc``),
    atom style, and reads in the structure data file.

    Args:
        structure (Atoms): The ASE structure whose PBC flags determine the boundary conditions.

    Returns:
        list[str]: LAMMPS input command strings (each ending with ``\\n``) to be written
            at the top of an input file.
    """
    dimension = 3
    boundary = " ".join(["p" if coord else "f" for coord in structure.pbc])
    init_commands = [
        "units metal\n",
        "dimension " + str(dimension) + "\n",
        "boundary " + boundary + "\n",
        "atom_style atomic\n",
        "read_data lammps.data\n",
    ]
    return init_commands


def _write_lammps_input_file(
    working_directory: str,
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    input_template: str,
) -> None:
    """
    Write the LAMMPS structure data file and input script to ``working_directory``.

    The structure is written as ``lammps.data`` and the input script as ``lmp.in``.
    The script is assembled from the initialisation commands, the potential ``Config``
    lines, the dump commands, and the rendered ``input_template``.

    Args:
        working_directory (str): Directory in which to write the LAMMPS input files.
        structure (Atoms): The ASE structure to write.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        input_template (str): Pre-rendered LAMMPS commands appended after the potential section.
    """
    _write_lammps_structure(
        structure=structure,
        potential_elements=potential_dataframe["Species"],
        bond_dict=None,
        units="metal",
        file_name="lammps.data",
        working_directory=working_directory,
    )
    input_str = (
        "".join(_lammps_file_initialization(structure=structure))
        + "\n".join(potential_dataframe["Config"])
        + "\n"
        + "".join(DUMP_COMMANDS)
        + input_template
    )
    with open(os.path.join(working_directory, "lmp.in"), "w") as f:
        f.writelines(input_str)


def optimize_positions_and_volume_with_lammpsfile(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    working_directory: str,
    executable_function: Callable[[str], Any],
    min_style: str = "cg",
    etol: float = 0.0,
    ftol: float = 0.0001,
    maxiter: int = 100000,
    maxeval: int = 10000000,
    thermo: int = 10,
    pressure: float | Iterable[float | None] = 0.0,
    vmax: Optional[float] = None,
) -> Atoms:
    """
    Relax atomic positions and cell with LAMMPS using file-based I/O.

    Writes LAMMPS input files, executes the calculator, parses the output, and
    returns a structure with the relaxed positions and cell.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        working_directory (str): Directory for LAMMPS input/output files.
        executable_function (Callable[[str], Any]): Callable that runs LAMMPS in the given directory.
        min_style (str): LAMMPS minimisation style (e.g. ``"cg"``). Defaults to ``"cg"``.
        etol (float): Energy tolerance for minimisation. Defaults to ``0.0``.
        ftol (float): Force tolerance for minimisation in eV/Å. Defaults to ``0.0001``.
        maxiter (int): Maximum number of minimisation iterations. Defaults to ``100000``.
        maxeval (int): Maximum number of force evaluations. Defaults to ``10000000``.
        thermo (int): Thermo output frequency. Defaults to ``10``.
        pressure (float | Iterable[float | None]): Target pressure for ``box/relax`` in bar.
        vmax (float | None): Maximum fractional volume change per step for ``box/relax``.

    Returns:
        Atoms: A copy of the input structure with relaxed positions and cell.
    """
    template_str = "\n".join(
        [
            get_box_relax_command(pressure=pressure, vmax=vmax),
            LAMMPS_THERMO_STYLE,
            LAMMPS_THERMO,
            LAMMPS_MINIMIZE,
        ]
    )
    input_template = Template(template_str).render(
        min_style=min_style,
        etol=etol,
        ftol=ftol,
        maxiter=maxiter,
        maxeval=maxeval,
        thermo=thermo,
    )
    _write_lammps_input_file(
        working_directory=working_directory,
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=input_template,
    )
    executable_function(working_directory)
    output = _parse_lammps_output_files(
        working_directory=working_directory,
        structure=structure,
        potential_elements=potential_dataframe["Species"],
        units="metal",
        prism=None,
        dump_h5_file_name="dump.h5",
        dump_out_file_name="dump.out",
        log_lammps_file_name="log.lammps",
    )
    structure_copy = structure.copy()
    structure_copy.set_cell(output["generic"]["cells"][-1], scale_atoms=True)
    structure_copy.positions = output["generic"]["positions"][-1]
    return structure_copy


def optimize_positions_with_lammpsfile(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    working_directory: str,
    executable_function: Callable[[str], Any],
    min_style: str = "cg",
    etol: float = 0.0,
    ftol: float = 0.0001,
    maxiter: int = 100000,
    maxeval: int = 10000000,
    thermo: int = 10,
) -> Atoms:
    """
    Relax atomic positions with LAMMPS using file-based I/O (cell fixed).

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        working_directory (str): Directory for LAMMPS input/output files.
        executable_function (Callable[[str], Any]): Callable that runs LAMMPS in the given directory.
        min_style (str): LAMMPS minimisation style. Defaults to ``"cg"``.
        etol (float): Energy tolerance for minimisation. Defaults to ``0.0``.
        ftol (float): Force tolerance for minimisation in eV/Å. Defaults to ``0.0001``.
        maxiter (int): Maximum number of minimisation iterations. Defaults to ``100000``.
        maxeval (int): Maximum number of force evaluations. Defaults to ``10000000``.
        thermo (int): Thermo output frequency. Defaults to ``10``.

    Returns:
        Atoms: A copy of the input structure with relaxed atomic positions.
    """
    template_str = "\n".join([LAMMPS_THERMO_STYLE, LAMMPS_THERMO, LAMMPS_MINIMIZE])
    input_template = Template(template_str).render(
        min_style=min_style,
        etol=etol,
        ftol=ftol,
        maxiter=maxiter,
        maxeval=maxeval,
        thermo=thermo,
    )
    _write_lammps_input_file(
        working_directory=working_directory,
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=input_template,
    )
    executable_function(working_directory)
    output = _parse_lammps_output_files(
        working_directory=working_directory,
        structure=structure,
        potential_elements=potential_dataframe["Species"],
        units="metal",
        prism=None,
        dump_h5_file_name="dump.h5",
        dump_out_file_name="dump.out",
        log_lammps_file_name="log.lammps",
    )
    structure_copy = structure.copy()
    structure_copy.positions = output["generic"]["positions"][-1]
    return structure_copy


def calc_static_with_lammpsfile(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    working_directory: str,
    executable_function: Callable[[str], Any],
    output_keys=OutputStatic.keys(),
) -> dict[str, Any]:
    """
    Run a static LAMMPS calculation using file-based I/O and return the requested output.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        working_directory (str): Directory for LAMMPS input/output files.
        executable_function (Callable[[str], Any]): Callable that runs LAMMPS in the given directory.
        output_keys: Which output quantities to return. Defaults to all ``OutputStatic`` keys.

    Returns:
        dict[str, Any]: Requested output quantities keyed by name.
    """
    template_str = "\n".join([LAMMPS_THERMO_STYLE, LAMMPS_THERMO, LAMMPS_RUN])
    input_template = Template(template_str).render(
        run=0,
        thermo=100,
    )
    _write_lammps_input_file(
        working_directory=working_directory,
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=input_template,
    )
    executable_function(working_directory)
    output = _parse_lammps_output_files(
        working_directory=working_directory,
        structure=structure,
        potential_elements=potential_dataframe["Species"],
        units="metal",
        prism=None,
        dump_h5_file_name="dump.h5",
        dump_out_file_name="dump.out",
        log_lammps_file_name="log.lammps",
    )
    output_obj = GenericOutput(output_dict=output)
    result_dict = OutputStatic(
        forces=output_obj.get_forces,
        energy=output_obj.get_energy_pot,
        stress=output_obj.get_stress,
        volume=output_obj.get_volume,
    ).get(output_keys=output_keys)
    return result_dict


@as_task_dict_evaluator
def evaluate_with_lammpsfile(
    structure: Atoms,
    tasks: list[str],
    potential_dataframe: pandas.DataFrame,
    working_directory: str,
    executable_function: Callable[[str], Any],
    lmp_optimizer_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Evaluate a task dictionary using LAMMPS file-based I/O and return results for all tasks.

    Dispatches to the appropriate file-calculator function based on the requested tasks.
    Decorated with ``as_task_dict_evaluator``.

    Args:
        structure (Atoms): The input structure.
        tasks (list[str]): List of task name strings.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        working_directory (str): Directory for LAMMPS input/output files.
        executable_function (Callable[[str], Any]): Callable that runs LAMMPS in the given directory.
        lmp_optimizer_kwargs (dict[str, Any] | None): Extra keyword arguments forwarded to the
            underlying optimisation or static functions.

    Returns:
        dict[str, Any]: Results keyed by output quantity name.

    Raises:
        ValueError: If none of the requested tasks are implemented by this calculator.
    """
    if lmp_optimizer_kwargs is None:
        lmp_optimizer_kwargs = {}
    results: dict[str, Any] = {}
    if "optimize_positions_and_volume" in tasks:
        results["structure_with_optimized_positions_and_volume"] = (
            optimize_positions_and_volume_with_lammpsfile(
                structure=structure,
                potential_dataframe=potential_dataframe,
                working_directory=working_directory,
                executable_function=executable_function,
                **lmp_optimizer_kwargs,
            )
        )
    elif "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = (
            optimize_positions_with_lammpsfile(
                structure=structure,
                potential_dataframe=potential_dataframe,
                working_directory=working_directory,
                executable_function=executable_function,
                **lmp_optimizer_kwargs,
            )
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_lammpsfile(
            structure=structure,
            potential_dataframe=potential_dataframe,
            working_directory=working_directory,
            executable_function=executable_function,
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The LAMMPS calculator does not implement:", tasks)
    return results
