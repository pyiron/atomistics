import os

from ase.atoms import Atoms
import pandas
from jinja2 import Template
from pyiron_lammps.structure import write_lammps_datafile
from pyiron_lammps.output import parse_lammps_output

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.lammps.commands import (
    LAMMPS_RUN,
    LAMMPS_THERMO,
    LAMMPS_THERMO_STYLE,
)
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputStatic


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
    dimension = 3
    boundary = " ".join(["p" if coord else "f" for coord in structure.pbc])
    init_commands = [
        "units metal\n",
        "dimension " + str(dimension) + "\n",
        "boundary " + boundary + "\n",
        "atom_style atomic\n",
        "read_data lammps.data\n",
    ]
    dump_commands = [
        "dump 1 all custom 100 dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
        'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
    ]
    input_str = "".join(init_commands) + "\n".join(potential_dataframe["Config"]) + "\n" + "".join(
        dump_commands) + input_template
    with open(os.path.join(working_directory, "lmp.in"), "w") as f:
        f.writelines(input_str)

    write_lammps_datafile(
        structure=structure,
        el_eam_lst=potential_dataframe["Species"],
        bond_dict=None,
        units='metal',
        file_name='lammps.data',
        cwd=working_directory,
    )

    print(executable_function(working_directory))
    output = parse_lammps_output(
        working_directory=working_directory,
        structure=structure,
        potential_elements=potential_dataframe["Species"],
        units="metal",
        prism=None,
        dump_h5_file_name='dump.h5',
        dump_out_file_name='dump.out',
        log_lammps_file_name='log.lammps',
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
) -> dict:
    if "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_lammpsfile(
            structure=structure,
            potential_dataframe=potential_dataframe,
            working_directory=working_directory,
            executable_function=executable_function,
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The LAMMPS calculator does not implement:", tasks)
