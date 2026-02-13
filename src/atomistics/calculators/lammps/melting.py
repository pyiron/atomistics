import numpy as np
import operator
import random
from ase.build import bulk
from ase.data import reference_states, atomic_numbers
from structuretoolkit.analyse import (
    get_adaptive_cna_descriptors,
    get_diamond_structure_descriptors,
)
from atomistics.shared.output import OutputMolecularDynamics
from atomistics.calculators.lammps import (
    optimize_positions_and_volume_with_lammpslib,
    calc_molecular_dynamics_npt_with_lammpslib,
)


def _check_diamond(structure):
    """
    Utility function to check if the structure is fcc, bcc, hcp or diamond

    Args:
        structure (pyiron_atomistics.structure.atoms.Atoms): Atomistic Structure object to check

    Returns:
        bool: true if diamond else false
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


def _analyse_structure(structure, mode="total", diamond=False):
    """
    Use either common neighbor analysis or the diamond structure detector

    Args:
        structure (pyiron_atomistics.structure.atoms.Atoms): The structure to analyze.
        mode ("total"/"numeric"/"str"): Controls the style and level
            of detail of the output.
            - total : return number of atoms belonging to each structure
            - numeric : return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - str : return a per atom string of sructures
        diamond (bool): Flag to either use the diamond structure detector or
            the common neighbor analysis.

    Returns:
        (depends on `mode`)
    """
    if not diamond:
        return get_adaptive_cna_descriptors(
            structure=structure, mode=mode, ovito_compatibility=True
        )
    else:
        return get_diamond_structure_descriptors(
            structure=structure, mode=mode, ovito_compatibility=True
        )


def _analyse_minimized_structure(structure):
    """

    Args:
        ham (GenericJob):

    Returns:

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


def _next_calc(structure, potential, temperature, seed, run_time_steps=10000):
    """
    Calculate NPT ensemble at a given temperature using the job defined in the project parameters:
    - job_type: Type of Simulation code to be used
    - project: Project object used to create the job
    - potential: Interatomic Potential
    - queue (optional): HPC Job queue to be used

    Args:
        structure (pyiron_atomistics.structure.atoms.Atoms): Atomistic Structure object to be set to the job as input sturcture
        temperature (float): Temperature of the Molecular dynamics calculation
        run_time_steps (int): Number of Molecular dynamics steps

    Returns:
        Final Atomistic Structure object
    """
    output_md_dict = calc_molecular_dynamics_npt_with_lammpslib(
        structure=structure,
        potential_dataframe=potential,
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


def _next_step_funct(
    number_of_atoms,
    key_max,
    structure_left,
    structure_right,
    potential,
    temperature_left,
    temperature_right,
    distribution_initial_half,
    structure_after_minimization,
    run_time_steps,
    crystalstructure,
    seed,
):
    """

    Args:
        number_of_atoms:
        key_max:
        structure_left:
        structure_right:
        temperature_left:
        temperature_right:
        distribution_initial_half:
        structure_after_minimization:
        run_time_steps:

    Returns:

    """
    structure_left_dict = _analyse_structure(
        structure=structure_left,
        mode="total",
        diamond=crystalstructure.lower() == "diamond",
    )
    structure_right_dict = _analyse_structure(
        structure=structure_right,
        mode="total",
        diamond=crystalstructure.lower() == "diamond",
    )
    temperature_diff = temperature_right - temperature_left
    if (
        structure_left_dict[key_max] / number_of_atoms > distribution_initial_half
        and structure_right_dict[key_max] / number_of_atoms > distribution_initial_half
    ):
        structure_left = structure_right.copy()
        temperature_left = temperature_right
        temperature_right += temperature_diff
        structure_right = _next_calc(
            structure=structure_after_minimization,
            temperature=temperature_right,
            potential=potential,
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
        structure_left = _next_calc(
            structure=structure_after_minimization,
            temperature=temperature_left,
            potential=potential,
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
        structure_left = _next_calc(
            structure=structure_after_minimization,
            temperature=temperature_left,
            potential=potential,
            seed=seed,
            run_time_steps=run_time_steps,
        )
    else:
        raise ValueError("We should never reach this point!")
    return structure_left, structure_right, temperature_left, temperature_right


def estimate_melting_temperature(
    element,
    potential,
    strain_run_time_steps=1000,
    temperature_left=0,
    temperature_right=1000,
    number_of_atoms=8000,
    seed=None,
):
    if seed is None:
        seed = random.randint(0, 99999)
    crystalstructure = reference_states[atomic_numbers[element]]["symmetry"]
    if crystalstructure == "hcp":
        basis = bulk(name=element, orthorhombic=True)
    else:
        basis = bulk(name=element, cubic=True)
    basis_lst = [basis.repeat([i, i, i]) for i in range(5, 30)]
    basis = basis_lst[
        np.argmin([np.abs(len(b) - number_of_atoms / 2) for b in basis_lst])
    ]

    structure_opt = optimize_positions_and_volume_with_lammpslib(
        structure=basis,
        potential_dataframe=potential,
        min_style="cg",
        etol=0.0,
        ftol=0.0001,
        maxiter=100000,
        maxeval=10000000,
        thermo=10,
        lmp=None,
    )

    (
        structure_after_minimization,
        key_max,
        number_of_atoms,
        distribution_initial_half,
        _,
    ) = _analyse_minimized_structure(structure=structure_opt)

    structure_left = structure_after_minimization
    structure_right = _next_calc(
        structure=structure_after_minimization,
        temperature=temperature_right,
        seed=seed,
        potential=potential,
        run_time_steps=strain_run_time_steps,
    )
    temperature_step = temperature_right - temperature_left

    while temperature_step > 10:
        (
            structure_left,
            structure_right,
            temperature_left,
            temperature_right,
        ) = _next_step_funct(
            number_of_atoms=number_of_atoms,
            key_max=key_max,
            structure_left=structure_left,
            structure_right=structure_right,
            potential=potential,
            temperature_left=temperature_left,
            temperature_right=temperature_right,
            distribution_initial_half=distribution_initial_half,
            structure_after_minimization=structure_after_minimization,
            run_time_steps=strain_run_time_steps,
            seed=seed,
            crystalstructure=crystalstructure,
        )
        temperature_step = temperature_right - temperature_left
    return int(round(temperature_left))
