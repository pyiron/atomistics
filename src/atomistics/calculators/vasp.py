import os
from collections.abc import Callable
from typing import Optional

from ase.atoms import Atoms
from ase.calculators.vasp.create_input import GenerateVaspInput
from vaspparser.vasp.output import parse_vasp_output

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputStatic


class OutputParser:
    """Parse VASP output files from a completed calculation.

    Args:
        working_directory (str): Path to the directory containing VASP output files.
        structure (Atoms): The input ASE structure used for the calculation.
    """

    def __init__(self, working_directory: str, structure: Atoms):
        self._output_dict = parse_vasp_output(
            working_directory=working_directory, structure=structure
        )

    def get_energy(self) -> float:
        """Return the total energy from the last ionic step in eV."""
        return self._output_dict["generic"]["energy_tot"][-1]

    def get_forces(self) -> list:
        """Return the atomic forces from the last ionic step in eV/Å."""
        return self._output_dict["generic"]["forces"][-1]

    def get_volume(self) -> float:
        """Return the cell volume from the last ionic step in Å³."""
        return self._output_dict["generic"]["volume"][-1]

    def get_stress(self) -> list:
        """Return the stress tensor from the last ionic step in kBar."""
        return self._output_dict["generic"]["stresses"][-1]


def write_input(working_directory: str, atoms: Atoms, **kwargs) -> None:
    """
    Write VASP input files (INCAR, KPOINTS, POSCAR, POTCAR) to a directory.

    Args:
        working_directory (str): Directory in which to write the input files (created if absent).
        atoms (Atoms): The ASE structure to use as the POSCAR.
        **kwargs: Additional VASP settings forwarded to ``GenerateVaspInput.set``.
    """
    vip = GenerateVaspInput()
    vip.set(**kwargs)
    vip.initialize(atoms=atoms)
    os.makedirs(working_directory, exist_ok=True)
    vip.write_input(atoms=atoms, directory=working_directory)


def calc_static_with_vasp(
    structure: Atoms,
    working_directory: str,
    executable_function: Callable[[str], object],
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    kpts: Optional[list[int]] = None,
    output_keys: list[str] | tuple[str, ...] = OutputStatic.keys(),
    **kwargs,
) -> dict:
    """
    Run a static VASP calculation and return the requested output quantities.

    Args:
        structure (Atoms): The input structure.
        working_directory (str): Directory used for VASP input/output files.
        executable_function (Callable[[str], object]): Callable that executes VASP in the given directory.
        prec (str): VASP precision tag (PREC). Defaults to ``"Accurate"``.
        algo (str): Electronic minimisation algorithm (ALGO). Defaults to ``"Fast"``.
        lreal (bool): Whether to use real-space projection (LREAL). Defaults to ``False``.
        lwave (bool): Whether to write the wave function file (LWAVE). Defaults to ``False``.
        lorbit (int): Controls orbital-decomposed DOS/PROCAR output (LORBIT). Defaults to ``0``.
        kpts (list[int] | None): Monkhorst-Pack k-point mesh. Defaults to ``[4, 4, 4]``.
        output_keys (list[str] | tuple[str, ...]): Which output quantities to return.
        **kwargs: Additional VASP settings forwarded to ``write_input``.

    Returns:
        dict: Requested output quantities keyed by name.
    """
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
    executable_function: Callable[[str], object],
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    isif: int = 2,
    ibrion: int = 2,
    nsw: int = 100,
    kpts: Optional[list[int]] = None,
    **kwargs,
) -> Atoms:
    """
    Relax atomic positions with VASP (cell shape and volume fixed).

    Args:
        structure (Atoms): The input structure.
        working_directory (str): Directory used for VASP input/output files.
        executable_function (Callable[[str], object]): Callable that executes VASP in the given directory.
        prec (str): VASP precision tag (PREC). Defaults to ``"Accurate"``.
        algo (str): Electronic minimisation algorithm (ALGO). Defaults to ``"Fast"``.
        lreal (bool): Whether to use real-space projection (LREAL). Defaults to ``False``.
        lwave (bool): Whether to write the wave function file (LWAVE). Defaults to ``False``.
        lorbit (int): Controls orbital-decomposed DOS/PROCAR output (LORBIT). Defaults to ``0``.
        isif (int): VASP ISIF tag controlling which degrees of freedom are relaxed. Defaults to ``2``.
        ibrion (int): VASP IBRION tag selecting the ionic relaxation algorithm. Defaults to ``2``.
        nsw (int): Maximum number of ionic steps (NSW). Defaults to ``100``.
        kpts (list[int] | None): Monkhorst-Pack k-point mesh. Defaults to ``[4, 4, 4]``.
        **kwargs: Additional VASP settings forwarded to ``write_input``.

    Returns:
        Atoms: A copy of the input structure with relaxed atomic positions.
    """
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
    executable_function: Callable[[str], object],
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    isif: int = 3,
    ibrion: int = 2,
    nsw: int = 100,
    kpts: Optional[list[int]] = None,
    **kwargs,
) -> Atoms:
    """
    Relax atomic positions and cell volume with VASP (cell shape fixed).

    Args:
        structure (Atoms): The input structure.
        working_directory (str): Directory used for VASP input/output files.
        executable_function (Callable[[str], object]): Callable that executes VASP in the given directory.
        prec (str): VASP precision tag (PREC). Defaults to ``"Accurate"``.
        algo (str): Electronic minimisation algorithm (ALGO). Defaults to ``"Fast"``.
        lreal (bool): Whether to use real-space projection (LREAL). Defaults to ``False``.
        lwave (bool): Whether to write the wave function file (LWAVE). Defaults to ``False``.
        lorbit (int): Controls orbital-decomposed DOS/PROCAR output (LORBIT). Defaults to ``0``.
        isif (int): VASP ISIF tag (3 = relax positions and volume). Defaults to ``3``.
        ibrion (int): VASP IBRION tag selecting the ionic relaxation algorithm. Defaults to ``2``.
        nsw (int): Maximum number of ionic steps (NSW). Defaults to ``100``.
        kpts (list[int] | None): Monkhorst-Pack k-point mesh. Defaults to ``[4, 4, 4]``.
        **kwargs: Additional VASP settings forwarded to ``write_input``.

    Returns:
        Atoms: A copy of the input structure with relaxed positions and scaled cell.
    """
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
    executable_function: Callable[[str], object],
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    isif: int = 7,
    ibrion: int = 2,
    nsw: int = 100,
    kpts: Optional[list[int]] = None,
    **kwargs,
) -> Atoms:
    """
    Relax cell volume with VASP (positions and cell shape fixed).

    Args:
        structure (Atoms): The input structure.
        working_directory (str): Directory used for VASP input/output files.
        executable_function (Callable[[str], object]): Callable that executes VASP in the given directory.
        prec (str): VASP precision tag (PREC). Defaults to ``"Accurate"``.
        algo (str): Electronic minimisation algorithm (ALGO). Defaults to ``"Fast"``.
        lreal (bool): Whether to use real-space projection (LREAL). Defaults to ``False``.
        lwave (bool): Whether to write the wave function file (LWAVE). Defaults to ``False``.
        lorbit (int): Controls orbital-decomposed DOS/PROCAR output (LORBIT). Defaults to ``0``.
        isif (int): VASP ISIF tag (7 = volume only). Defaults to ``7``.
        ibrion (int): VASP IBRION tag selecting the ionic relaxation algorithm. Defaults to ``2``.
        nsw (int): Maximum number of ionic steps (NSW). Defaults to ``100``.
        kpts (list[int] | None): Monkhorst-Pack k-point mesh. Defaults to ``[4, 4, 4]``.
        **kwargs: Additional VASP settings forwarded to ``write_input``.

    Returns:
        Atoms: A copy of the input structure with the cell scaled to the optimised volume.
    """
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
    executable_function: Callable[[str], object],
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    isif: int = 5,
    ibrion: int = 2,
    nsw: int = 100,
    kpts: Optional[list[int]] = None,
    **kwargs,
) -> Atoms:
    """
    Relax cell shape and volume with VASP (atomic positions fixed).

    Args:
        structure (Atoms): The input structure.
        working_directory (str): Directory used for VASP input/output files.
        executable_function (Callable[[str], object]): Callable that executes VASP in the given directory.
        prec (str): VASP precision tag (PREC). Defaults to ``"Accurate"``.
        algo (str): Electronic minimisation algorithm (ALGO). Defaults to ``"Fast"``.
        lreal (bool): Whether to use real-space projection (LREAL). Defaults to ``False``.
        lwave (bool): Whether to write the wave function file (LWAVE). Defaults to ``False``.
        lorbit (int): Controls orbital-decomposed DOS/PROCAR output (LORBIT). Defaults to ``0``.
        isif (int): VASP ISIF tag (5 = cell shape and volume, positions fixed). Defaults to ``5``.
        ibrion (int): VASP IBRION tag selecting the ionic relaxation algorithm. Defaults to ``2``.
        nsw (int): Maximum number of ionic steps (NSW). Defaults to ``100``.
        kpts (list[int] | None): Monkhorst-Pack k-point mesh. Defaults to ``[4, 4, 4]``.
        **kwargs: Additional VASP settings forwarded to ``write_input``.

    Returns:
        Atoms: A copy of the input structure with the relaxed cell.
    """
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
    executable_function: Callable[[str], object],
    prec: str = "Accurate",
    algo: str = "Fast",
    lreal: bool = False,
    lwave: bool = False,
    lorbit: int = 0,
    kpts: Optional[list[int]] = None,
    **kwargs,
) -> dict:
    """
    Evaluate a task dictionary using VASP and return results for all requested tasks.

    Dispatches to the appropriate VASP calculation function based on the tasks requested.
    Decorated with ``as_task_dict_evaluator`` so it accepts a task dict and returns a
    results dict rather than operating on a single structure.

    Args:
        structure (Atoms): The input structure.
        tasks (list): List of task name strings (e.g. ``["calc_energy", "calc_forces"]``).
        working_directory (str): Directory used for VASP input/output files.
        executable_function (Callable[[str], object]): Callable that executes VASP in the given directory.
        prec (str): VASP precision tag (PREC). Defaults to ``"Accurate"``.
        algo (str): Electronic minimisation algorithm (ALGO). Defaults to ``"Fast"``.
        lreal (bool): Whether to use real-space projection (LREAL). Defaults to ``False``.
        lwave (bool): Whether to write the wave function file (LWAVE). Defaults to ``False``.
        lorbit (int): Controls orbital-decomposed DOS/PROCAR output (LORBIT). Defaults to ``0``.
        kpts (list[int] | None): Monkhorst-Pack k-point mesh. Defaults to ``[4, 4, 4]``.
        **kwargs: Additional VASP settings forwarded to the underlying calculation functions.

    Returns:
        dict: Results keyed by output quantity name.

    Raises:
        ValueError: If none of the requested tasks are implemented by this calculator.
    """
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
