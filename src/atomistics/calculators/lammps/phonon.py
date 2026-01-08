import sys

import dynaphopy.dynamics as dyn
import numpy as np
import pandas
from ase.atoms import Atoms
from dynaphopy.atoms import Structure
from dynaphopy.interface.iofile import get_correct_arrangement
from dynaphopy.interface.phonopy_link import ForceConstants
from dynaphopy.power_spectrum import _progress_bar

from atomistics.calculators.lammps.helpers import lammps_run


def generate_pylammps_trajectory(
    structure: Atoms,
    lmp,
    total_time: float = 0.1,  # picoseconds
    time_step: float = 0.002,  # picoseconds
    relaxation_time: int = 0,  # in timesteps
    silent: bool = False,
    memmap: bool = False,  # not fully implemented yet!
    velocity_only: bool = False,
    temperature: float = None,
    thermostat_mass: float = 0.5,
    sampling_interval: int = 1,  # in timesteps
):
    lmp.interactive_lib_command("neighbor 0.3 bin")
    lmp.interactive_lib_command(f"timestep {time_step}")

    # Force reset temperature (overwrites lammps script)
    # This forces NVT simulation
    if temperature is not None:
        lmp.interactive_lib_command(
            f"fix int all nvt temp {temperature} {temperature} {thermostat_mass}"
        )

    # Check if correct number of atoms
    if lmp._interactive_library.extract_global("natoms", 0) < 2:
        print("Number of atoms in MD should be higher than 1!")
        sys.exit()

    # Check if initial velocities all zero
    if not np.array(lmp._interactive_library.gather_atoms("v", 1, 3)).any():
        t = temperature if temperature is not None else 100
        lmp.interactive_lib_command(
            f"velocity all create {t} 3627941 dist gaussian mom yes"
        )
        lmp.interactive_lib_command(f"velocity all scale {t}")

    lmp.interactive_lib_command("run 0")
    simulation_cell = lmp.interactive_cells_getter()

    positions = []
    velocity = []
    energy = []

    reference = lmp.interactive_positions_getter()
    template = get_correct_arrangement(reference, structure)
    indexing = np.argsort(template)

    lmp.interactive_lib_command(f"run {int(relaxation_time / time_step)}")

    if not silent:
        _progress_bar(0, "lammps")

    n_loops = int(total_time / time_step / sampling_interval)
    for i in range(n_loops):
        if not silent:
            _progress_bar(
                float((i + 1) * time_step * sampling_interval) / total_time,
                "lammps",
            )

        lmp.interactive_lib_command(f"run {sampling_interval}")
        energy.append(lmp.interactive_energy_pot_getter())
        velocity.append(lmp.interactive_velocities_getter()[indexing, :])

        if not velocity_only:
            positions.append(lmp.interactive_positions_getter()[indexing, :])

    positions = np.array(positions, dtype=complex)
    velocity = np.array(velocity, dtype=complex)
    energy = np.array(energy)

    if velocity_only:
        positions = None

    lmp.close()

    time = np.array(
        [i * time_step * sampling_interval for i in range(n_loops)], dtype=float
    )
    return structure, positions, velocity, time, energy, simulation_cell, memmap


def calc_molecular_dynamics_phonons_with_lammpslib(
    structure_ase: Atoms,
    potential_dataframe: pandas.DataFrame,
    force_constants: np.ndarray,
    phonopy_unitcell,
    phonopy_primitive_matrix,
    phonopy_supercell_matrix,
    total_time: int = 20,  # ps
    time_step: float = 0.001,  # ps
    relaxation_time: int = 5,  # ps
    silent: bool = True,
    supercell: list[int] = None,
    memmap: bool = False,
    velocity_only: bool = True,
    temperature: float = 300.0,
):
    if supercell is None:
        supercell = [2, 2, 2]
    dp_structure = Structure(
        cell=phonopy_unitcell.cell,
        scaled_positions=phonopy_unitcell.scaled_positions,
        atomic_elements=phonopy_unitcell.symbols,
        primitive_matrix=phonopy_primitive_matrix,
        force_constants=ForceConstants(
            force_constants,
            supercell=phonopy_supercell_matrix,
        ),
    )
    structure_ase_repeated = structure_ase.repeat(supercell)
    lmp_instance = lammps_run(
        structure=structure_ase_repeated,
        potential_dataframe=potential_dataframe,
        input_template=None,
        lmp=None,
        disable_log_file=False,
        working_directory=".",
    )
    (
        structure,
        positions,
        velocity,
        time,
        energy,
        simulation_cell,
        memmap,
    ) = generate_pylammps_trajectory(
        structure=dp_structure,
        lmp=lmp_instance,
        total_time=total_time,
        time_step=time_step,
        relaxation_time=relaxation_time,
        silent=silent,
        memmap=memmap,
        velocity_only=velocity_only,
        temperature=temperature,
    )
    return dyn.Dynamics(
        structure=structure,
        trajectory=positions,
        velocity=velocity,
        time=time,
        energy=energy,
        supercell=simulation_cell,
        memmap=memmap,
    )
