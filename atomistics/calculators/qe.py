import os
import subprocess

import numpy as np
from ase.atoms import Atoms
from ase.io import write
from pwtools import io

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputStatic


class QEStaticParser:
    """
    Class for parsing Quantum Espresso static calculations output file.
    """

    def __init__(self, filename: str):
        """
        Initialize the QEStaticParser.

        Args:
            filename (str): The path to the Quantum Espresso output file.
        """
        self.parser = io.read_pw_scf(filename=filename, use_alat=True)

    def forces(self) -> np.ndarray:
        """
        Get the forces from the calculation.

        Returns:
            np.ndarray: The forces array.
        """
        return self.parser.forces

    def energy(self) -> float:
        """
        Get the total energy from the calculation.

        Returns:
            float: The total energy.
        """
        return self.parser.etot

    def stress(self) -> np.ndarray:
        """
        Get the stress tensor from the calculation.

        Returns:
            np.ndarray: The stress tensor.
        """
        return self.parser.stress

    def volume(self) -> float:
        """
        Get the volume from the calculation.

        Returns:
            float: The volume.
        """
        return self.parser.volume


def call_qe_via_ase_command(calculation_name: str, working_directory: str) -> None:
    """
    Call Quantum Espresso via ASE command.

    Args:
        calculation_name (str): The name of the calculation.
        working_directory (str): The working directory.

    Returns:
        None
    """
    subprocess.check_output(
        os.environ["ASE_ESPRESSO_COMMAND"].replace("PREFIX", calculation_name),
        shell=True,
        universal_newlines=True,
        cwd=working_directory,
    )


def set_pseudo_potentials(pseudopotentials: dict, structure: Atoms) -> dict:
    """
    Set the pseudopotentials for the given structure.

    Args:
        pseudopotentials (dict): A dictionary mapping element symbols to pseudopotential filenames.
        structure (Atoms): The atomic structure.

    Returns:
        dict: A dictionary mapping element symbols to pseudopotential filenames.
    """
    if pseudopotentials is not None:
        return pseudopotentials
    else:
        pseudopotentials_base = {
            "Ag": "Ag_ONCV_PBE-1.0.oncvpsp.upf",
            "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
            "Ar": "Ar_ONCV_PBE-1.1.oncvpsp.upf",
            "As": "As.pbe-n-rrkjus_psl.0.2.UPF",
            "Au": "Au_ONCV_PBE-1.0.oncvpsp.upf",
            "B": "B_pbe_v1.01.uspp.F.UPF",
            "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Be": "Be_ONCV_PBE-1.0.oncvpsp.upf",
            "Bi": "Bi_pbe_v1.uspp.F.UPF",
            "Br": "br_pbe_v1.4.uspp.F.UPF",
            "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
            "Ca": "Ca_pbe_v1.uspp.F.UPF",
            "Cd": "Cd.pbe-dn-rrkjus_psl.0.3.1.UPF",
            "Ce": "Ce.GGA-PBE-paw-v1.0.UPF",
            "Cl": "Cl.pbe-n-rrkjus_psl.1.0.0.UPF",
            "Co": "Co_pbe_v1.2.uspp.F.UPF",
            "Cr": "cr_pbe_v1.5.uspp.F.UPF",
            "Cs": "Cs_pbe_v1.uspp.F.UPF",
            "Cu": "Cu_ONCV_PBE-1.0.oncvpsp.upf",
            "Dy": "Dy.GGA-PBE-paw-v1.0.UPF",
            "Er": "Er.GGA-PBE-paw-v1.0.UPF",
            "Eu": "Eu.GGA-PBE-paw-v1.0.UPF",
            "F": "F.oncvpsp.upf",
            "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
            "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "Gd": "Gd.GGA-PBE-paw-v1.0.UPF",
            "Ge": "ge_pbe_v1.4.uspp.F.UPF",
            "H": "H_ONCV_PBE-1.0.oncvpsp.upf",
            "He": "He_ONCV_PBE-1.0.oncvpsp.upf",
            "Hf": "Hf-sp.oncvpsp.upf",
            "Hg": "Hg_ONCV_PBE-1.0.oncvpsp.upf",
            "Ho": "Ho.GGA-PBE-paw-v1.0.UPF",
            "I": "I.pbe-n-kjpaw_psl.0.2.UPF",
            "In": "In.pbe-dn-rrkjus_psl.0.2.2.UPF",
            "Ir": "Ir_pbe_v1.2.uspp.F.UPF",
            "K": "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Kr": "Kr_ONCV_PBE-1.0.oncvpsp.upf",
            "La": "La.GGA-PBE-paw-v1.0.UPF",
            "Li": "li_pbe_v1.4.uspp.F.UPF",
            "Lu": "Lu.GGA-PBE-paw-v1.0.UPF",
            "Mg": "mg_pbe_v1.4.uspp.F.UPF",
            "Mn": "mn_pbe_v1.5.uspp.F.UPF",
            "Mo": "Mo_ONCV_PBE-1.0.oncvpsp.upf",
            "N": "N.oncvpsp.upf",
            "Na": "Na_ONCV_PBE-1.0.oncvpsp.upf",
            "Nb": "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF",
            "Nd": "Nd.GGA-PBE-paw-v1.0.UPF",
            "Ne": "Ne_ONCV_PBE-1.0.oncvpsp.upf",
            "Ni": "ni_pbe_v1.4.uspp.F.UPF",
            "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
            "Os": "Os_pbe_v1.2.uspp.F.UPF",
            "P": "P.pbe-n-rrkjus_psl.1.0.0.UPF",
            "Pb": "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
            "Pd": "Pd_ONCV_PBE-1.0.oncvpsp.upf",
            "Pm": "Pm.GGA-PBE-paw-v1.0.UPF",
            "Po": "Po.pbe-dn-rrkjus_psl.1.0.0.UPF",
            "Pr": "Pr.GGA-PBE-paw-v1.0.UPF",
            "Pt": "Pt.pbe-spfn-rrkjus_psl.1.0.0.UPF",
            "Rb": "Rb_ONCV_PBE-1.0.oncvpsp.upf",
            "Re": "Re_pbe_v1.2.uspp.F.UPF",
            "Rh": "Rh_ONCV_PBE-1.0.oncvpsp.upf",
            "Rn": "Rn.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "Ru": "Ru_ONCV_PBE-1.0.oncvpsp.upf",
            "S": "s_pbe_v1.4.uspp.F.UPF",
            "Sb": "sb_pbe_v1.4.uspp.F.UPF",
            "Sc": "Sc.pbe-spn-kjpaw_psl.0.2.3.UPF",
            "Se": "Se_pbe_v1.uspp.F.UPF",
            "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
            "Sm": "Sm.GGA-PBE-paw-v1.0.UPF",
            "Sn": "Sn_pbe_v1.uspp.F.UPF",
            "Sr": "Sr_pbe_v1.uspp.F.UPF",
            "Ta": "Ta_pbe_v1.uspp.F.UPF",
            "Tb": "Tb.GGA-PBE-paw-v1.0.UPF",
            "Tc": "Tc_ONCV_PBE-1.0.oncvpsp.upf",
            "Te": "Te_pbe_v1.uspp.F.UPF",
            "Ti": "ti_pbe_v1.4.uspp.F.UPF",
            "Tl": "Tl_pbe_v1.2.uspp.F.UPF",
            "Tm": "Tm.GGA-PBE-paw-v1.0.UPF",
            "V": "v_pbe_v1.4.uspp.F.UPF",
            "W": "W_pbe_v1.2.uspp.F.UPF",
            "Xe": "Xe_ONCV_PBE-1.1.oncvpsp.upf",
            "Y": "Y_pbe_v1.uspp.F.UPF",
            "Yb": "Yb.GGA-PBE-paw-v1.0.UPF",
            "Zn": "Zn_pbe_v1.uspp.F.UPF",
            "Zr": "Zr_pbe_v1.uspp.F.UPF",
        }
        return {
            pseudopotentials_base[el]
            for el in list(set(structure.get_chemical_symbols()))
        }


def generate_input_data(**kwargs) -> dict:
    """
    Generate input data for Quantum Espresso calculation.

    Args:
        **kwargs: Additional keyword arguments.

    Returns:
        dict: The input data dictionary.
    """
    return kwargs


def optimize_positions_and_volume_with_qe(
    structure: Atoms,
    calculation_name: str = "espresso",
    working_directory: str = ".",
    kpts: tuple[int] = (3, 3, 3),
    pseudopotentials: dict = None,
    tstress: bool = True,
    tprnfor: bool = True,
    **kwargs,
) -> Atoms:
    """
    Optimize the positions and volume of a structure using Quantum Espresso.

    Args:
        structure (Atoms): The atomic structure.
        calculation_name (str, optional): The name of the calculation. Defaults to "espresso".
        working_directory (str, optional): The working directory. Defaults to ".".
        kpts (tuple[int], optional): The k-points for the calculation. Defaults to (3, 3, 3).
        pseudopotentials (dict, optional): A dictionary mapping element symbols to pseudopotential filenames. Defaults to None.
        tstress (bool, optional): Whether to calculate the stress. Defaults to True.
        tprnfor (bool, optional): Whether to print forces. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        Atoms: The optimized atomic structure.
    """
    input_file_name = os.path.join(working_directory, calculation_name + ".pwi")
    output_file_name = os.path.join(working_directory, calculation_name + ".pwo")
    input_data = generate_input_data(
        calculation="vc-relax",
        cell_dofree="ibrav",
        **kwargs,
    )
    pseudopotentials = set_pseudo_potentials(
        pseudopotentials=pseudopotentials,
        structure=structure,
    )
    write(
        input_file_name,
        structure,
        Crystal=True,
        kpts=kpts,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        tstress=tstress,
        tprnfor=tprnfor,
    )
    call_qe_via_ase_command(
        calculation_name=calculation_name, working_directory=working_directory
    )
    return io.read_pw_md(output_file_name)[-1].get_ase_atoms()


def calc_static_with_qe(
    structure: Atoms,
    calculation_name: str = "espresso",
    working_directory: str = ".",
    kpts: tuple[int] = (3, 3, 3),
    pseudopotentials: dict = None,
    tstress: bool = True,
    tprnfor: bool = True,
    output_keys: tuple[str] = OutputStatic.keys(),
    **kwargs,
) -> OutputStatic:
    """
    Calculate static properties of a structure using Quantum Espresso.

    Args:
        structure (Atoms): The atomic structure.
        calculation_name (str, optional): The name of the calculation. Defaults to "espresso".
        working_directory (str, optional): The working directory. Defaults to ".".
        kpts (tuple[int], optional): The k-points for the calculation. Defaults to (3, 3, 3).
        pseudopotentials (dict, optional): A dictionary mapping element symbols to pseudopotential filenames. Defaults to None.
        tstress (bool, optional): Whether to calculate the stress. Defaults to True.
        tprnfor (bool, optional): Whether to print forces. Defaults to True.
        output_keys (tuple[str], optional): The keys of the output properties to retrieve. Defaults to OutputStatic.keys().
        **kwargs: Additional keyword arguments.

    Returns:
        OutputStatic: The static output properties.
    """
    input_file_name = os.path.join(working_directory, calculation_name + ".pwi")
    output_file_name = os.path.join(working_directory, calculation_name + ".pwo")
    os.makedirs(working_directory, exist_ok=True)
    input_data = generate_input_data(
        calculation="scf",
        **kwargs,
    )
    pseudopotentials = set_pseudo_potentials(
        pseudopotentials=pseudopotentials, structure=structure
    )
    write(
        input_file_name,
        structure,
        Crystal=True,
        kpts=kpts,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        tstress=tstress,
        tprnfor=tprnfor,
    )
    call_qe_via_ase_command(
        calculation_name=calculation_name, working_directory=working_directory
    )
    parser = QEStaticParser(filename=output_file_name)
    return OutputStatic(**{k: getattr(parser, k) for k in OutputStatic.keys()}).get(
        output_keys=output_keys
    )


@as_task_dict_evaluator
def evaluate_with_qe(
    structure: Atoms,
    tasks: dict,
    calculation_name: str = "espresso",
    working_directory: str = ".",
    kpts: tuple[int] = (3, 3, 3),
    pseudopotentials: dict = None,
    tstress: bool = True,
    tprnfor: bool = True,
    **kwargs,
) -> dict:
    """
    Evaluate tasks using Quantum Espresso.

    Args:
        structure (Atoms): The atomic structure.
        tasks (dict): The tasks to evaluate.
        calculation_name (str, optional): The name of the calculation. Defaults to "espresso".
        working_directory (str, optional): The working directory. Defaults to ".".
        kpts (tuple[int], optional): The k-points for the calculation. Defaults to (3, 3, 3).
        pseudopotentials (dict, optional): A dictionary mapping element symbols to pseudopotential filenames. Defaults to None.
        tstress (bool, optional): Whether to calculate the stress. Defaults to True.
        tprnfor (bool, optional): Whether to print forces. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: The results of the evaluated tasks.
    """
    results = {}
    if "optimize_positions_and_volume" in tasks:
        results["structure_with_optimized_positions_and_volume"] = (
            optimize_positions_and_volume_with_qe(
                structure=structure,
                calculation_name=calculation_name,
                working_directory=working_directory,
                kpts=kpts,
                pseudopotentials=pseudopotentials,
                tstress=tstress,
                tprnfor=tprnfor,
                **kwargs,
            )
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        results = calc_static_with_qe(
            structure=structure,
            calculation_name=calculation_name,
            working_directory=working_directory,
            kpts=kpts,
            pseudopotentials=pseudopotentials,
            tstress=tstress,
            tprnfor=tprnfor,
            output_keys=get_quantities_from_tasks(tasks=tasks),
            **kwargs,
        )
    else:
        raise ValueError("The Quantum Espresso calculator does not implement:", tasks)
    return results
