import operator
import random
import re

from ase import Atoms
from ase.build import bulk
from ase.data import reference_states, atomic_numbers
from atomistics.shared.output import OutputMolecularDynamics
from atomistics.calculators.lammps import (
    optimize_positions_and_volume_with_lammpslib,
    calc_molecular_dynamics_npt_with_lammpslib,
)
import numpy as np
import pandas as pd
from structuretoolkit.analyse import (
    get_adaptive_cna_descriptors,
    get_diamond_structure_descriptors,
)

def _check_diamond(structure: Atoms) -> bool:
    """
    Check if the structure is diamond or not by comparing the counts of 'OTHER' from
    adaptive CNA and diamond structure descriptors.

    Args:
        structure (Atoms): The structure to check.

    Returns:
        bool: True if the structure is diamond, False otherwise.
    """

    cna_dict = get_adaptive_cna_descriptors(
        structure=structure, mode="total", ovito_compatibility=True
    )
    dia_dict = get_diamond_structure_descriptors(
        structure=structure, mode="total", ovito_compatibility=True
    )
    return (
        cna_dict["CommonNeighborAnalysis.counts.OTHER"]
        > dia_dict["IdentifyDiamond.counts.OTHER"]
    )

def _analyse_structure(
    structure: Atoms, mode: str = "total", diamond: bool = False
) -> dict:
    """
    Analyse the structure using either adaptive CNA or diamond structure descriptors.

    Args:
        structure (Atoms): The structure to analyse.
        mode (str): The mode for analysis, default is "total".
        diamond (bool): Flag to choose between diamond structure descriptors or adaptive CNA.

    Returns:
        dict: A dictionary of structure descriptors.
    """
    if not diamond:
        return get_adaptive_cna_descriptors(
            structure=structure, mode=mode, ovito_compatibility=True
        )
    else:
        return get_diamond_structure_descriptors(
            structure=structure, mode=mode, ovito_compatibility=True
        )

def _analyse_minimized_structure(structure: Atoms, diamond_flag: bool) -> tuple:
    """

    Args:
        structure (Atoms): The minimized structure.

    Returns:
        tuple: A tuple containing the structure, key of the maximum structure type,
               number of atoms, half of the initial distribution of the identified structure type,
               and the final structure dictionary.

    """
    final_structure_dict = _analyse_structure(
        structure=structure, mode="total", diamond=diamond_flag
    )
    key_max = max(final_structure_dict.items(), key=operator.itemgetter(1))[0]
    number_of_atoms = len(structure)
    distribution_initial = final_structure_dict[key_max] / number_of_atoms
    distribution_initial_half = distribution_initial / 2

    return (
        structure,
        key_max,
        number_of_atoms,
        distribution_initial_half,
        final_structure_dict,
    )

def _get_repeated_structure(structure: Atoms, target_number_of_atoms: int) -> Atoms:
    """
    Get a repeated structure that is as close as possible to the target number of atoms.

    Args:
        structure (Atoms): The input structure to be repeated.
        target_number_of_atoms (int): The target number of atoms for the simulation cell.

    Returns:
        Atoms: The repeated structure.
    """
    r_est = (target_number_of_atoms / len(structure)) ** (1 / 3)
    candidates = np.array(
        [max(1, int(np.floor(r_est))), int(np.round(r_est)), int(np.ceil(r_est))]
    )
    basis_lst = [structure.repeat([i, i, i]) for i in candidates]
    basis = basis_lst[
        np.argmin([np.abs(len(b) - target_number_of_atoms) for b in basis_lst])
    ]

    return basis

def _run_npt_molecular_dynamics(
    structure: Atoms,
    potential_dataframe: pd.DataFrame,
    temperature: float,
    seed: int,
    run_time_steps: int = 10000,
) -> Atoms:
    """
    Calculate NPT ensemble at a given temperature using the job defined in the project parameters:
    - job_type: Type of Simulation code to be used
    - project: Project object used to create the job
    - potential_dataframe: Interatomic Potential dataframe
    - queue (optional): HPC Job queue to be used

    Args:
        structure (pyiron_atomistics.structure.atoms.Atoms): Atomistic Structure object to be set to the job as input sturcture
        potential_dataframe (pd.DataFrame): Interatomic Potential dataframe
        temperature (float): Temperature of the Molecular dynamics calculation
        seed (int): Random seed for the simulation
        run_time_steps (int): Number of Molecular dynamics steps

    Returns:
        Final Atomistic Structure object
    """
    output_md_dict = calc_molecular_dynamics_npt_with_lammpslib(
        structure=structure,
        potential_dataframe=potential_dataframe,
        Tstart=temperature,
        Tstop=temperature,
        Tdamp=0.1,
        run=run_time_steps,
        thermo=1000,
        timestep=0.001,
        Pstart=0.0,
        Pstop=0.0,
        Pdamp=1.0,
        seed=seed,
        dist="gaussian",
        velocity_rescale_factor=1.0,
        lmp=None,
        output_keys=OutputMolecularDynamics.keys(),
    )
    structure_md = structure.copy()
    structure_md.set_positions(output_md_dict["positions"][-1])
    structure_md.set_cell(output_md_dict["cell"][-1])

    return structure_md

def _run_next_bisection_iteration(
    number_of_atoms: int,
    key_max: str,
    structure_left: Atoms,
    structure_right: Atoms,
    potential_dataframe: pd.DataFrame,
    temperature_left: float,
    temperature_right: float,
    distribution_initial_half: float,
    structure_after_minimization: Atoms,
    run_time_steps: int,
    diamond_flag: bool,
    seed: int,
) -> tuple[Atoms, Atoms, float, float]:
    """
    Run the next iteration of the bisection method for estimating the melting temperature.

    Args:
        number_of_atoms: int
        key_max: str
        structure_left: Atoms
        structure_right: Atoms
        temperature_left: float
        temperature_right: float
        distribution_initial_half: float
        structure_after_minimization: Atoms
        run_time_steps: int
        diamond_flag: bool
        seed: int

    Returns:
        tuple[Atoms, Atoms, float, float]: A tuple containing the updated left and right structures,
                                            and the updated left and right temperatures.
    """
    structure_left_dict = _analyse_structure(
        structure=structure_left,
        mode="total",
        diamond=diamond_flag,
    )
    structure_right_dict = _analyse_structure(
        structure=structure_right,
        mode="total",
        diamond=diamond_flag,
    )
    temperature_diff = temperature_right - temperature_left
    if (
        structure_left_dict[key_max] / number_of_atoms > distribution_initial_half
        and structure_right_dict[key_max] / number_of_atoms > distribution_initial_half
    ):
        structure_left = structure_right.copy()
        temperature_left = temperature_right
        temperature_right += temperature_diff
        structure_right = _run_npt_molecular_dynamics(
            structure=structure_after_minimization,
            temperature=temperature_right,
            potential_dataframe=potential_dataframe,
            seed=seed,
            run_time_steps=run_time_steps,
        )
    elif (
        structure_left_dict[key_max] / number_of_atoms
        > distribution_initial_half
        > structure_right_dict[key_max] / number_of_atoms
    ):
        temperature_diff /= 2
        temperature_left += temperature_diff
        structure_left = _run_npt_molecular_dynamics(
            structure=structure_after_minimization,
            temperature=temperature_left,
            potential_dataframe=potential_dataframe,
            seed=seed,
            run_time_steps=run_time_steps,
        )
    elif (
        structure_left_dict[key_max] / number_of_atoms < distribution_initial_half
        and structure_right_dict[key_max] / number_of_atoms < distribution_initial_half
    ):
        temperature_diff /= 2
        temperature_right = temperature_left
        temperature_left -= temperature_diff
        structure_right = structure_left.copy()
        structure_left = _run_npt_molecular_dynamics(
            structure=structure_after_minimization,
            temperature=temperature_left,
            potential_dataframe=potential_dataframe,
            seed=seed,
            run_time_steps=run_time_steps,
        )
    else:
        raise ValueError("We should never reach this point!")
    
    return structure_left, structure_right, temperature_left, temperature_right


def estimate_melting_temperature_using_bisection_CNA(
    structure: Atoms,
    potential_dataframe: pd.DataFrame,
    target_number_of_atoms: int,
    strain_run_time_steps: int = 1000,
    temperature_left: float = 0,
    temperature_right: float = 1000,
    number_of_atoms: int = 8000,
    seed: int = None,
):

    if seed is None:
        seed = random.randint(0, 99999)

    diamond_flag = _check_diamond(structure=structure)
    repeated_structure = _get_repeated_structure(
        structure=structure, target_number_of_atoms=target_number_of_atoms
    )

    position_and_volume_optimized_structure = (
        optimize_positions_and_volume_with_lammpslib(
            structure=repeated_structure,
            potential_dataframe=potential_dataframe,
            min_style="cg",
            etol=0.0,
            ftol=0.0001,
            maxiter=100000,
            maxeval=10000000,
            thermo=10,
            lmp=None,
        )
    )

    (
        structure_after_minimization,
        key_max,
        number_of_atoms,
        distribution_initial_half,
        _,
    ) = _analyse_minimized_structure(
        structure=position_and_volume_optimized_structure, diamond_flag=diamond_flag
    )

    structure_left = structure_after_minimization
    structure_right = _run_npt_molecular_dynamics(
        structure=structure_after_minimization,
        temperature=temperature_right,
        seed=seed,
        potential_dataframe=potential_dataframe,
        run_time_steps=strain_run_time_steps,
    )
    temperature_step = temperature_right - temperature_left

    while temperature_step > 10:
        (
            structure_left,
            structure_right,
            temperature_left,
            temperature_right,
        ) = _run_next_bisection_iteration(
            number_of_atoms=number_of_atoms,
            key_max=key_max,
            structure_left=structure_left,
            structure_right=structure_right,
            potential_dataframe=potential_dataframe,
            temperature_left=temperature_left,
            temperature_right=temperature_right,
            distribution_initial_half=distribution_initial_half,
            structure_after_minimization=structure_after_minimization,
            run_time_steps=strain_run_time_steps,
            seed=seed,
            diamond_flag=diamond_flag,
        )
        temperature_step = temperature_right - temperature_left

    return int(round(temperature_left))