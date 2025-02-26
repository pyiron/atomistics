import os

import pandas
from ase.atoms import Atoms
from jinja2 import Template
from pyiron_lammps import parse_lammps_output_files as _parse_lammps_output_files
from pyiron_lammps import write_lammps_structure as _write_lammps_structure

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.lammps.commands import (
    LAMMPS_MINIMIZE,
    LAMMPS_MINIMIZE_VOLUME,
    LAMMPS_RUN,
    LAMMPS_THERMO,
    LAMMPS_THERMO_STYLE,
)
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputStatic

DUMP_COMMANDS = [
    "dump 1 all custom 100 dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
    'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
]


class GenericOutput:
    def __init__(self, output_dict):
        self._output_dict = output_dict

    def get_forces(self):
        return self._output_dict["generic"]["forces"][-1]

    def get_energy_pot(self):
        return self._output_dict["generic"]["energy_pot"][-1]

    def get_stress(self):
        return self._output_dict["generic"]["pressures"][-1]

    def get_volume(self):
        return self._output_dict["generic"]["volume"][-1]


def _lammps_file_initialization(structure):
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
    working_directory, structure, potential_dataframe, input_template
):
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
    executable_function: callable,
    min_style: str = "cg",
    etol: float = 0.0,
    ftol: float = 0.0001,
    maxiter: int = 100000,
    maxeval: int = 10000000,
    thermo: int = 10,
) -> Atoms:
    template_str = (
        LAMMPS_MINIMIZE_VOLUME
        + "\n"
        + LAMMPS_THERMO_STYLE
        + "\n"
        + LAMMPS_THERMO
        + "\n"
        + LAMMPS_MINIMIZE
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
    executable_function: callable,
    min_style: str = "cg",
    etol: float = 0.0,
    ftol: float = 0.0001,
    maxiter: int = 100000,
    maxeval: int = 10000000,
    thermo: int = 10,
) -> Atoms:
    template_str = LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_MINIMIZE
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
    executable_function: callable,
    output_keys=OutputStatic.keys(),
) -> dict:
    template_str = LAMMPS_THERMO_STYLE + "\n" + LAMMPS_THERMO + "\n" + LAMMPS_RUN
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
    tasks: list,
    potential_dataframe: pandas.DataFrame,
    working_directory: str,
    executable_function: callable,
    lmp_optimizer_kwargs: dict = None,
) -> dict:
    if lmp_optimizer_kwargs is None:
        lmp_optimizer_kwargs = {}
    results = {}
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
