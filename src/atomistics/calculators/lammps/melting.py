import operator
import numpy as np
from ase import Atoms
from random import random
from typing import Optional
from pandas import DataFrame

from structuretoolkit.analyse import (
    get_adaptive_cna_descriptors,
    get_diamond_structure_descriptors,
)

from atomistics.calculators.lammps import (
    optimize_positions_and_volume_with_lammpslib,
    calc_molecular_dynamics_npt_with_lammpslib,
)


def _run_npt_molecular_dynamics(
    structure: Atoms,
    potential_dataframe: DataFrame,
    temperature: float,
    run: int = 10000,
    seed: int = 42,
    cores: int = 1,
    log_file: Optional[str] = None,
) -> Atoms:
    """
    Run NPT molecular dynamics at a given temperature. Initialize velocities at the
    given temperature and couple stresses in all x, y, and z directions.

    Args:
        structure (Atoms): The input structure for MD simulation.
        potential_dataframe (DataFrame): The dataframe containing the potential information.
        temperature (float): The temperature at which to run the MD simulation.
        run (int): The number of time steps for the MD simulation.
        seed (int): The random seed for velocity initialization.
        cores (int): The number of CPU cores to use.
        log_file (Optional[str]): The log file path.

    Returns:
        Atoms: The structure after MD simulation.
    """

    print(f"Running NPT MD at T = {temperature} K")
    output_md_dict = calc_molecular_dynamics_npt_with_lammpslib(
        structure=structure,
        potential_dataframe=potential_dataframe,
        Tstart=temperature,
        Tstop=temperature,
        run=run,
        seed=seed,
        velocity_rescale_factor=1.0,
        couple_xyz=True,
        cores=cores,
        log_file=log_file,
    )
    structure_md = structure.copy()
    structure_md.set_positions(output_md_dict["positions"][-1])
    structure_md.set_cell(output_md_dict["cell"][-1])
    return structure_md


def _get_repeated_structure(
    input_structure: Atoms, target_number_of_atoms: int
) -> Atoms:
    """
    Get a repeated structure that is as close as possible to the target number of atoms.

    Args:
        input_structure (Atoms): The input structure to be repeated.
        target_number_of_atoms (int): The target number of atoms for the simulation cell.

    Returns:
        Atoms: The repeated structure.
    """

    r_est = (target_number_of_atoms / len(input_structure)) ** (1 / 3)
    candidates = np.array(
        [max(1, int(np.floor(r_est))), int(np.round(r_est)), int(np.ceil(r_est))]
    )
    basis_lst = [input_structure.repeat([i, i, i]) for i in candidates]
    basis = basis_lst[
        np.argmin([np.abs(len(b) - target_number_of_atoms) for b in basis_lst])
    ]

    return basis


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


def _analyse_minimized_structure(structure: Atoms) -> tuple:
    """
    Analyse the minimized structure to get initial parameters for the bisection algorithm.

    Args:
        structure (Atoms): The minimized structure.

    Returns:
        tuple: A tuple containing the structure, key of the maximum structure type,
               number of atoms, half of the initial distribution of the identified structure type,
               and the final structure dictionary.
    """

    diamond_flag = _check_diamond(structure=structure)
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


def _run_next_bisection_iteration(
    potential_dataframe: DataFrame,
    diamond_flag: bool,
    number_of_atoms: int,
    key_max: str,
    structure_left: Atoms,
    structure_right: Atoms,
    temperature_left: float,
    temperature_right: float,
    distribution_initial_half: float,
    structure_after_minimization: Atoms,
    run: int = 1000,
    seed: int = 42,
    cores: int = 1,
    log_file: Optional[str] = None,
) -> tuple:
    """
    Run the next iteration of the bisection algorithm. This function updates the left and right structures.
    It checks the structure types at the left and right temperatures and decides how to update the
    temperature bounds based on the distribution of the identified structure from adaptive CNA or diamond analysis.

    Args:
        potential_dataframe (DataFrame): The dataframe containing the potential information.
        diamond_flag (bool): Flag to either use the diamond structure detector or the common neighbor analysis.
        number_of_atoms (int): The number of atoms in the structure.
        key_max (str): The structure type with the maximum count from the initial analysis.
        structure_left (Atoms): The structure at the left temperature bound.
        structure_right (Atoms): The structure at the right temperature bound.
        temperature_left (float): The left temperature bound.
        temperature_right (float): The right temperature bound.
        distribution_initial_half (float): Half of the initial distribution of the identified structure type.
        structure_after_minimization (Atoms): The structure after position and volume optimization.
        run (int): The number of time steps for each MD simulation.
        seed (int): The random seed for velocity initialization.
        cores (int): The number of CPU cores to use.
        log_file (Optional[str]): The log file path.

    Returns:
        tuple: Updated structures and temperature bounds.
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
        print(
            f"Both structures have {key_max} above half distribution at temperatures {temperature_left} and {temperature_right}"
        )
        print("Increasing temperatures...")
        print("-----------------------------------------------------")
        structure_left = structure_right.copy()
        temperature_left = temperature_right
        temperature_right += temperature_diff
        structure_right = _run_npt_molecular_dynamics(
            structure=structure_after_minimization,
            potential_dataframe=potential_dataframe,
            temperature=temperature_right,
            run=run,
            seed=seed,
            cores=cores,
            log_file=log_file,
        )
    elif (
        structure_left_dict[key_max] / number_of_atoms
        > distribution_initial_half
        > structure_right_dict[key_max] / number_of_atoms
    ):
        print(
            f"Left structure has {key_max} above half distribution at temperature {temperature_left}, right structure below at temperature {temperature_right}"
        )
        print("Decreasing temperatures...")
        print("-----------------------------------------------------")
        temperature_diff /= 2
        temperature_left += temperature_diff
        structure_left = _run_npt_molecular_dynamics(
            structure=structure_after_minimization,
            potential_dataframe=potential_dataframe,
            temperature=temperature_left,
            run=run,
            seed=seed,
            cores=cores,
            log_file=log_file,
        )
    elif (
        structure_left_dict[key_max] / number_of_atoms < distribution_initial_half
        and structure_right_dict[key_max] / number_of_atoms < distribution_initial_half
    ):
        print(
            f"Both structures have {key_max} below half distribution at temperatures {temperature_left} and {temperature_right}"
        )
        print("Decreasing temperatures...")
        print("-----------------------------------------------------")
        temperature_diff /= 2
        temperature_right = temperature_left
        temperature_left -= temperature_diff
        structure_right = structure_left.copy()
        structure_left = _run_npt_molecular_dynamics(
            structure=structure_after_minimization,
            potential_dataframe=potential_dataframe,
            temperature=temperature_left,
            run=run,
            seed=seed,
            cores=cores,
            log_file=log_file,
        )
    else:
        raise ValueError("We should never reach this point!")

    return structure_left, structure_right, temperature_left, temperature_right


def _run_bisection_algorithm(
    optimized_structure: Atoms,
    potential_dataframe: DataFrame,
    temperature_left: float,
    temperature_right: float,
    temperature_diff_tolerance: float = 10,
    run: int = 10000,
    seed: int = 42,
    cores: int = 1,
    log_file: Optional[str] = None,
) -> float:
    """
    Run the bisection algorithm to estimate the melting temperature.

    Args:
        optimized_structure (Atoms): The optimized structure after position and volume optimization.
        potential_dataframe (DataFrame): The dataframe containing the potential information.
        temperature_left (float): The lower bound of the temperature range.
        temperature_right (float): The upper bound of the temperature range.
        temperature_diff_tolerance (float): The tolerance for the temperature difference to stop the bisection.
        run (int): The number of time steps for each MD simulation.
        seed (int): The random seed for velocity initialization.
        cores (int): The number of CPU cores to use.
        log_file (Optional[str]): The log file path.

    Returns:
        float: The estimated melting temperature.
    """

    temperature_step = temperature_right - temperature_left

    (
        structure_after_minimization,
        key_max,
        number_of_atoms,
        distribution_initial_half,
        _,
    ) = _analyse_minimized_structure(structure=optimized_structure)

    diamond_flag = _check_diamond(structure=optimized_structure)

    # FIXME: Here the t right is being checked for, but seems like t low is assumed to be 0, and hence not checked for
    structure_left = optimized_structure

    print("Running at the highest temperature")
    structure_right = _run_npt_molecular_dynamics(
        structure=optimized_structure,
        potential_dataframe=potential_dataframe,
        temperature=temperature_right,
        run=run,
        seed=seed,
        cores=cores,
        log_file=log_file,
    )

    while temperature_step > temperature_diff_tolerance:
        (
            structure_left,
            structure_right,
            temperature_left,
            temperature_right,
        ) = _run_next_bisection_iteration(
            potential_dataframe=potential_dataframe,
            diamond_flag=diamond_flag,
            number_of_atoms=number_of_atoms,
            key_max=key_max,
            structure_left=structure_left,
            structure_right=structure_right,
            temperature_left=temperature_left,
            temperature_right=temperature_right,
            distribution_initial_half=distribution_initial_half,
            structure_after_minimization=structure_after_minimization,
            run=run,
            seed=seed,
            cores=cores,
            log_file=log_file,
        )
        temperature_step = temperature_right - temperature_left

    print(f"Melting point estimated at T = {temperature_left} K")
    return temperature_left


def estimate_melting_temperature_using_bisection_CNA(
    structure: Atoms,
    potential_dataframe: DataFrame,
    target_number_of_atoms: int = 4000,
    temperature_left: float = 0,
    temperature_right: float = 1000,
    temperature_diff_tolerance: float = 10,
    run: int = 10000,
    seed: Optional[int] = None,
    cores: int = 1,
    log_file: Optional[str] = None,
) -> float:
    """
    Estimate the melting temperature of a given element using the bisection algorithm,
    and adaptive Common Neighbour Analysis (CNA).

    Args:
        structure (Atoms): The input structure of the element.
        potential_dataframe (DataFrame): The dataframe containing the potential information.
        target_number_of_atoms (int): The target number of atoms for the simulation cell.
        temperature_left (float): The lower bound of the temperature range.
        temperature_right (float): The upper bound of the temperature range.
        temperature_diff_tolerance (float): The tolerance for the temperature difference to stop the bisection.
        run (int): The number of time steps for each MD simulation.
        seed (Optional[int]): The random seed for velocity initialization.
        cores (int): The number of CPU cores to use.
        log_file (Optional[str]): The log file path.

    Returns:
        float: The estimated melting temperature.
    """

    if seed is None:
        seed = random.randint(0, 99999)
    repeated_structure = _get_repeated_structure(
        input_structure=structure, target_number_of_atoms=target_number_of_atoms
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
            cores=cores,
            log_file=log_file,
        )
    )

    temperature_final = _run_bisection_algorithm(
        optimized_structure=position_and_volume_optimized_structure,
        potential_dataframe=potential_dataframe,
        temperature_left=temperature_left,
        temperature_right=temperature_right,
        temperature_diff_tolerance=temperature_diff_tolerance,
        run=run,
        seed=seed,
        cores=cores,
        log_file=log_file,
    )

    return int(round(temperature_final))
