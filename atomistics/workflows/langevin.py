import numpy as np
from ase.atoms import Atoms
from scipy.constants import physical_constants

from atomistics.workflows.interface import Workflow

KB = physical_constants["Boltzmann constant in eV/K"][0]
EV_TO_U_ANGSQ_PER_FSSQ = physical_constants["Faraday constant"][0] / 10**7
U_ANGSQ_PER_FSSQ_TO_EV = 1.0 / EV_TO_U_ANGSQ_PER_FSSQ


def langevin_delta_v(
    temperature: float,
    time_step: float,
    masses: np.ndarray,
    velocities: np.ndarray,
    damping_timescale: float = None,
) -> np.ndarray:
    """
    Velocity changes due to the Langevin thermostat.

    Args:
        temperature (float): The target temperature in K.
        time_step (float): The MD time step in fs.
        masses (numpy.ndarray): Per-atom masses in u with a shape (N_atoms, 1).
        damping_timescale (float): The characteristic timescale of the thermostat in fs.
        velocities (numpy.ndarray): Per-atom velocities in angstrom/fs.

    Returns:
        (numpy.ndarray): Per atom accelerations to use for changing velocities.
    """
    if damping_timescale is not None:
        drag = -0.5 * time_step * velocities / damping_timescale
        noise = np.sqrt(
            EV_TO_U_ANGSQ_PER_FSSQ
            * KB
            * temperature
            * time_step
            / (masses * damping_timescale)
        ) * np.random.randn(*velocities.shape)
        noise -= np.mean(noise, axis=0)
        return drag + noise
    else:
        return 0.0


def convert_to_acceleration(forces: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Convert forces to accelerations.

    Args:
        forces (numpy.ndarray): Per-atom forces in eV/angstrom.
        masses (numpy.ndarray): Per-atom masses in u.

    Returns:
        (numpy.ndarray): Per-atom accelerations in angstrom/fs^2.
    """
    return forces * EV_TO_U_ANGSQ_PER_FSSQ / masses


def get_initial_velocities(
    temperature: float, masses: np.ndarray, overheat_fraction: float = 2.0
) -> np.ndarray:
    """
    Generate initial velocities for the Langevin thermostat.

    Args:
        temperature (float): The target temperature in K.
        masses (numpy.ndarray): Per-atom masses in u with a shape (N_atoms, 1).
        overheat_fraction (float): The factor to overheat the system by (default: 2.0).

    Returns:
        (numpy.ndarray): Per-atom velocities in angstrom/fs.
    """
    vel_scale = np.sqrt(EV_TO_U_ANGSQ_PER_FSSQ * KB * temperature / masses) * np.sqrt(
        overheat_fraction
    )
    vel_dir = np.random.randn(len(masses), 3)
    velocities = vel_scale * vel_dir
    velocities -= np.mean(velocities, axis=0)
    return velocities


def get_first_half_step(
    forces: np.ndarray, masses: np.ndarray, time_step: float, velocities: np.ndarray
) -> np.ndarray:
    """
    Calculate the velocities at the first half step of the Langevin workflow.

    Args:
        forces (numpy.ndarray): Per-atom forces in eV/angstrom.
        masses (numpy.ndarray): Per-atom masses in u.
        time_step (float): The MD time step in fs.
        velocities (numpy.ndarray): Per-atom velocities in angstrom/fs.

    Returns:
        (numpy.ndarray): Per-atom velocities at the first half step in angstrom/fs.
    """
    acceleration = convert_to_acceleration(forces, masses)
    return velocities + 0.5 * acceleration * time_step


class LangevinWorkflow(Workflow):
    """
    LangevinWorkflow class represents a workflow for performing Langevin dynamics simulation.

    Args:
        structure (ase.Atoms): The atomic structure.
        temperature (float, optional): The temperature in Kelvin. Default is 1000.0.
        overheat_fraction (float, optional): The fraction by which to overheat the system. Default is 2.0.
        damping_timescale (float, optional): The damping timescale in fs. Default is 100.0.
        time_step (int, optional): The time step in fs. Default is 1.

    Attributes:
        structure (ase.Atoms): The atomic structure.
        temperature (float): The temperature in Kelvin.
        overheat_fraction (float): The fraction by which the system is overheated.
        damping_timescale (float): The damping timescale in fs.
        time_step (int): The time step in fs.
        masses (numpy.ndarray): The masses of the atoms.
        positions (numpy.ndarray): The positions of the atoms.
        velocities (numpy.ndarray): The velocities of the atoms.
        gamma (numpy.ndarray): The damping coefficients of the atoms.
        forces (numpy.ndarray): The forces on the atoms.

    Methods:
        generate_structures: Generates the structures for the Langevin dynamics simulation.
        analyse_structures: Analyzes the structures generated in the Langevin dynamics simulation.

    """

    def __init__(
        self,
        structure: Atoms,
        temperature: float = 1000.0,
        overheat_fraction: float = 2.0,
        damping_timescale: float = 100.0,
        time_step: int = 1,
    ):
        self.structure = structure
        self.temperature = temperature
        self.overheat_fraction = overheat_fraction
        self.damping_timescale = damping_timescale
        self.time_step = time_step
        self.masses = np.array([a.mass for a in self.structure[:]])[:, np.newaxis]
        self.positions = self.structure.positions
        self.velocities = get_initial_velocities(
            temperature=self.temperature,
            masses=self.masses,
            overheat_fraction=self.overheat_fraction,
        )
        self.gamma = self.masses / self.damping_timescale
        self.forces = None

    def generate_structures(self) -> dict[str, dict[int, Atoms]]:
        """
        Generates the structures for the Langevin dynamics simulation.

        Returns:
            dict: A dictionary containing the generated structures.
        """
        if self.forces is not None:
            # first half step
            vel_half = get_first_half_step(
                forces=self.forces,
                masses=self.masses,
                time_step=self.time_step,
                velocities=self.velocities,
            )

            # damping
            vel_half += langevin_delta_v(
                temperature=self.temperature,
                time_step=self.time_step,
                masses=self.masses,
                damping_timescale=self.damping_timescale,
                velocities=self.velocities,
            )

            # postion update
            pos_step = self.positions + vel_half * self.time_step
            structure = self.structure.copy()
            structure.positions = pos_step
        else:
            structure = self.structure
        return {"calc_forces": {0: structure}, "calc_energy": {0: structure}}

    def analyse_structures(self, output_dict: dict[str, dict[int, Atoms]]):
        """
        Analyzes the structures generated in the Langevin dynamics simulation.

        Args:
            output_dict (dict): A dictionary containing the output structures.

        Returns:
            tuple: A tuple containing the potential energy and kinetic energy.
        """
        self.forces, eng_pot = output_dict["forces"][0], output_dict["energy"][0]

        # second half step
        acceleration = convert_to_acceleration(forces=self.forces, masses=self.masses)
        vel_step = self.velocities + 0.5 * acceleration * self.time_step

        # damping
        vel_step += langevin_delta_v(
            temperature=self.temperature,
            time_step=self.time_step,
            masses=self.masses,
            damping_timescale=self.damping_timescale,
            velocities=self.velocities,
        )

        # kinetic energy
        kinetic_energy = (
            0.5 * np.sum(self.masses * vel_step * vel_step) / EV_TO_U_ANGSQ_PER_FSSQ
        )
        self.velocities = vel_step.copy()
        return eng_pot, kinetic_energy
