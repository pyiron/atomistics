import os
from typing import Optional

from ase.atoms import Atoms
from ase.calculators.vasp.create_input import GenerateVaspInput
from pyiron_vasp.vasp.output import parse_vasp_output

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputStatic


class OutputParser:
    def __init__(self, working_directory, structure):
        self._output_dict = parse_vasp_output(
            working_directory=working_directory, structure=structure
        )

    def get_energy(self):
        return self._output_dict["generic"]["energy_tot"][-1]

    def get_forces(self):
        return self._output_dict["generic"]["forces"][-1]

    def get_volume(self):
        return self._output_dict["generic"]["volume"][-1]

    def get_stress(self):
        return self._output_dict["generic"]["stresses"][-1]


def write_input(working_directory, atoms, **kwargs):
    vip = GenerateVaspInput()
    vip.set(**kwargs)
    vip.initialize(atoms=atoms)
    os.makedirs(working_directory, exist_ok=True)
    vip.write_input(atoms=atoms, directory=working_directory)


def calc_static_with_vasp(
    structure: Atoms,
    working_directory: str,
    executable_function: callable,
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    kpts: Optional[list[int, int, int]] = None,
    output_keys: dict = OutputStatic.keys(),
    **kwargs,
) -> dict:
    if kpts is None:
        kpts = [4, 4, 4]
    write_input(
        working_directory=working_directory,
        atoms=structure,
        prec=prec,
        algo=algo,
        lreal=lreal,
        lwave=lwave,
        lorbit=lorbit,
        kpts=kpts,
        **kwargs,
    )
    executable_function(working_directory)
    output_obj = OutputParser(working_directory=working_directory, structure=structure)
    result_dict = OutputStatic(
        forces=output_obj.get_forces,
        energy=output_obj.get_energy,
        stress=output_obj.get_stress,
        volume=output_obj.get_volume,
    ).get(output_keys=output_keys)
    return result_dict


def optimize_positions_with_vasp(
    structure: Atoms,
    working_directory: str,
    executable_function: callable,
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    isif: int = 2,
    ibrion: int = 2,
    nsw: int = 100,
    kpts: Optional[list[int, int, int]] = None,
    **kwargs,
) -> Atoms:
    if kpts is None:
        kpts = [4, 4, 4]
    write_input(
        working_directory=working_directory,
        atoms=structure,
        prec=prec,
        algo=algo,
        lreal=lreal,
        lwave=lwave,
        lorbit=lorbit,
        kpts=kpts,
        isif=isif,
        ibrion=ibrion,
        nsw=nsw,
        **kwargs,
    )
    executable_function(working_directory)
    output_dict = parse_vasp_output(
        working_directory=working_directory, structure=structure
    )
    structure_copy = structure.copy()
    structure_copy.positions = output_dict["generic"]["positions"][-1]
    return structure_copy


def optimize_positions_and_volume_with_vasp(
    structure: Atoms,
    working_directory: str,
    executable_function: callable,
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    isif: int = 3,
    ibrion: int = 2,
    nsw: int = 100,
    kpts: Optional[list[int, int, int]] = None,
    **kwargs,
) -> Atoms:
    if kpts is None:
        kpts = [4, 4, 4]
    write_input(
        working_directory=working_directory,
        atoms=structure,
        prec=prec,
        algo=algo,
        lreal=lreal,
        lwave=lwave,
        lorbit=lorbit,
        kpts=kpts,
        isif=isif,
        ibrion=ibrion,
        nsw=nsw,
        **kwargs,
    )
    executable_function(working_directory)
    output_dict = parse_vasp_output(
        working_directory=working_directory, structure=structure
    )
    structure_copy = structure.copy()
    structure_copy.set_cell(output_dict["generic"]["cells"][-1], scale_atoms=True)
    structure_copy.positions = output_dict["generic"]["positions"][-1]
    return structure_copy


def optimize_volume_with_vasp(
    structure: Atoms,
    working_directory: str,
    executable_function: callable,
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    isif: int = 7,
    ibrion: int = 2,
    nsw: int = 100,
    kpts: Optional[list[int, int, int]] = None,
    **kwargs,
) -> Atoms:
    if kpts is None:
        kpts = [4, 4, 4]
    write_input(
        working_directory=working_directory,
        atoms=structure,
        prec=prec,
        algo=algo,
        lreal=lreal,
        lwave=lwave,
        lorbit=lorbit,
        kpts=kpts,
        isif=isif,
        ibrion=ibrion,
        nsw=nsw,
        **kwargs,
    )
    executable_function(working_directory)
    output_dict = parse_vasp_output(
        working_directory=working_directory, structure=structure
    )
    structure_copy = structure.copy()
    structure_copy.set_cell(output_dict["generic"]["cells"][-1], scale_atoms=True)
    return structure_copy


def optimize_cell_with_vasp(
    structure: Atoms,
    working_directory: str,
    executable_function: callable,
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    isif: int = 5,
    ibrion: int = 2,
    nsw: int = 100,
    kpts: Optional[list[int, int, int]] = None,
    **kwargs,
) -> Atoms:
    if kpts is None:
        kpts = [4, 4, 4]
    write_input(
        working_directory=working_directory,
        atoms=structure,
        prec=prec,
        algo=algo,
        lreal=lreal,
        lwave=lwave,
        lorbit=lorbit,
        kpts=kpts,
        isif=isif,
        ibrion=ibrion,
        nsw=nsw,
        **kwargs,
    )
    executable_function(working_directory)
    output_dict = parse_vasp_output(
        working_directory=working_directory, structure=structure
    )
    structure_copy = structure.copy()
    structure_copy.set_cell(output_dict["generic"]["cells"][-1], scale_atoms=True)
    return structure_copy


@as_task_dict_evaluator
def evaluate_with_vasp(
    structure: Atoms,
    tasks: list,
    working_directory: str,
    executable_function: callable,
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    kpts: Optional[list[int, int, int]] = None,
    **kwargs,
) -> dict:
    if kpts is None:
        kpts = [4, 4, 4]
    results = {}
    if "optimize_cell" in tasks:
        results["structure_with_optimized_cell"] = optimize_cell_with_vasp(
            structure=structure,
            working_directory=working_directory,
            executable_function=executable_function,
            prec=prec,
            algo=algo,
            lreal=lreal,
            lwave=lwave,
            lorbit=lorbit,
            kpts=kpts,
            **kwargs,
        )
    elif "optimize_positions_and_volume" in tasks:
        results["structure_with_optimized_positions_and_volume"] = (
            optimize_positions_and_volume_with_vasp(
                structure=structure,
                working_directory=working_directory,
                executable_function=executable_function,
                prec=prec,
                algo=algo,
                lreal=lreal,
                lwave=lwave,
                lorbit=lorbit,
                kpts=kpts,
                **kwargs,
            )
        )
    elif "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = optimize_positions_with_vasp(
            structure=structure,
            working_directory=working_directory,
            executable_function=executable_function,
            prec=prec,
            algo=algo,
            lreal=lreal,
            lwave=lwave,
            lorbit=lorbit,
            kpts=kpts,
            **kwargs,
        )
    elif "optimize_volume" in tasks:
        results["structure_with_optimized_volume"] = optimize_volume_with_vasp(
            structure=structure,
            working_directory=working_directory,
            executable_function=executable_function,
            prec=prec,
            algo=algo,
            lreal=lreal,
            lwave=lwave,
            lorbit=lorbit,
            kpts=kpts,
            **kwargs,
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_vasp(
            structure=structure,
            working_directory=working_directory,
            executable_function=executable_function,
            prec=prec,
            algo=algo,
            lreal=lreal,
            lwave=lwave,
            lorbit=lorbit,
            kpts=kpts,
            output_keys=get_quantities_from_tasks(tasks=tasks),
            **kwargs,
        )
    else:
        raise ValueError("The VASP calculator does not implement:", tasks)
    return results
