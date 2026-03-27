from __future__ import annotations

import numpy as np
from ase import units
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import FixAtoms
from ase.filters import Filter, UnitCellFilter
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize.optimize import Optimizer

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import (
    OutputMolecularDynamics,
    OutputStatic,
    OutputThermalExpansion,
)
from atomistics.shared.thermal_expansion import get_thermal_expansion_output
from atomistics.shared.tqdm_iterator import get_tqdm_iterator


class ASEExecutor:
    def __init__(self, ase_structure: Atoms, ase_calculator: ASECalculator) -> None:
        """
        Initialize the ASEExecutor.

        Args:
            ase_structure (Atoms): The ASE structure object.
            ase_calculator (ASECalculator): The ASE calculator object.
        """
        self.structure = ase_structure
        self.structure.calc = ase_calculator

    def forces(self) -> np.ndarray:
        """
        Get the forces on the atoms.

        Returns:
            np.ndarray: The forces on the atoms.
        """
        return self.structure.get_forces()

    def energy(self) -> float:
        """
        Get the potential energy of the system.

        Returns:
            float: The potential energy of the system.
        """
        return self.structure.get_potential_energy()

    def energy_pot(self) -> float:
        """
        Get the potential energy of the system.

        Returns:
            float: The potential energy of the system.
        """
        return self.structure.get_potential_energy()

    def energy_tot(self) -> float:
        """
        Get the total energy of the system.

        Returns:
            float: The total energy of the system.
        """
        return (
            self.structure.get_potential_energy() + self.structure.get_kinetic_energy()
        )

    def stress(self) -> np.ndarray:
        """
        Get the stress tensor of the system.

        Returns:
            np.ndarray: The stress tensor of the system.
        """
        try:
            return self.structure.get_stress(voigt=False)
        except PropertyNotImplementedError:
            return None

    def pressure(self) -> np.ndarray:
        """
        Get the pressure of the system.

        Returns:
            np.ndarray: The pressure of the system.
        """
        try:
            return self.structure.get_stress(voigt=False)
        except PropertyNotImplementedError:
            return None

    def cell(self) -> np.ndarray:
        """
        Get the cell parameters of the system.

        Returns:
            np.ndarray: The cell parameters of the system.
        """
        return self.structure.get_cell()

    def positions(self) -> np.ndarray:
        """
        Get the atomic positions of the system.

        Returns:
            np.ndarray: The atomic positions of the system.
        """
        return self.structure.get_positions()

    def velocities(self) -> np.ndarray:
        """
        Get the atomic velocities of the system.

        Returns:
            np.ndarray: The atomic velocities of the system.
        """
        return self.structure.get_velocities()

    def temperature(self) -> float:
        """
        Get the temperature of the system.

        Returns:
            float: The temperature of the system.
        """
        return self.structure.get_temperature()

    def volume(self) -> float:
        """
        Get the volume of the system.

        Returns:
            float: The volume of the system.
        """
        return self.structure.get_volume()


@as_task_dict_evaluator
def evaluate_with_ase(
    structure: Atoms,
    tasks: list,
    ase_calculator: ASECalculator,
    ase_optimizer: type[Optimizer] = None,
    ase_optimizer_kwargs: dict = None,
    filter_class: type[Filter] = UnitCellFilter,
) -> dict:
    """
    Evaluate tasks using ASE calculator.

    Args:
        structure (Atoms): The ASE structure object.
        tasks (list): List of tasks to evaluate.
        ase_calculator (ASECalculator): The ASE calculator object.
        ase_optimizer (Type[Optimizer], optional): The ASE optimizer object. Defaults to None.
        ase_optimizer_kwargs (dict, optional): Keyword arguments for the ASE optimizer. Defaults to {}.
        filter_class (Type[Filter]): The ASE filter class to use for filtering during structure optimization.

    Returns:
        dict: Dictionary containing the results of the evaluated tasks.
    """
    if ase_optimizer_kwargs is None:
        ase_optimizer_kwargs = {}
    results = {}
    if "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = optimize_positions_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            ase_optimizer=ase_optimizer,
            ase_optimizer_kwargs=ase_optimizer_kwargs,
        )
    elif "optimize_positions_and_volume" in tasks:
        results["structure_with_optimized_positions_and_volume"] = (
            optimize_positions_and_volume_with_ase(
                structure=structure,
                ase_calculator=ase_calculator,
                ase_optimizer=ase_optimizer,
                ase_optimizer_kwargs=ase_optimizer_kwargs,
                filter_class=filter_class,
            )
        )
    elif "optimize_volume" in tasks:
        results["structure_with_optimized_volume"] = optimize_volume_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            ase_optimizer=ase_optimizer,
            ase_optimizer_kwargs=ase_optimizer_kwargs,
            filter_class=filter_class,
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_ase(
            structure=structure,
            ase_calculator=ase_calculator,
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The ASE calculator does not implement:", tasks)
    return results


def calc_static_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    output_keys: list[str] = OutputStatic.keys(),
) -> dict:
    """
    Calculate static properties using ASE calculator.

    Args:
        structure (Atoms): The ASE structure object.
        ase_calculator (ASECalculator): The ASE calculator object.
        output_keys (list[str], optional): List of output keys. Defaults to OutputStatic.keys().

    Returns:
        dict: Dictionary containing the calculated static properties.
    """
    ase_exe = ASEExecutor(ase_structure=structure, ase_calculator=ase_calculator)
    return OutputStatic(**{k: getattr(ase_exe, k) for k in OutputStatic.keys()}).get(
        output_keys=output_keys
    )


def calc_molecular_dynamics_langevin_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 1.0,
    temperature: float = 100.0,
    friction: float = 0.002,
    output_keys: list[str] = OutputMolecularDynamics.keys(),
) -> dict:
    """
    Perform molecular dynamics simulation using the Langevin algorithm with ASE.

    Args:
        structure (Atoms): The atomic structure to simulate.
        ase_calculator (ASECalculator): The ASE calculator to use for energy and force calculations.
        run (int): The number of MD steps to perform.
        thermo (int): The interval at which to print thermodynamic properties.
        timestep (float): The time step size in fs.
        temperature (float): The desired temperature in Kelvin.
        friction (float): The friction coefficient for the Langevin thermostat.
        output_keys (list[str]): The keys of the properties to include in the output dictionary.

    Returns:
        dict: A dictionary containing the calculated properties at each MD step.
    """
    return _calc_molecular_dynamics_with_ase(
        dyn=Langevin(
            atoms=structure,
            timestep=timestep,
            temperature_K=temperature,
            friction=friction,
        ),
        structure=structure,
        ase_calculator=ase_calculator,
        temperature=temperature,
        run=run,
        thermo=thermo,
        output_keys=output_keys,
    )


def optimize_positions_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    ase_optimizer: type[Optimizer],
    ase_optimizer_kwargs: dict,
) -> Atoms:
    """
    Optimize the atomic positions of the structure using ASE optimizer.

    Args:
        structure (Atoms): The ASE structure object.
        ase_calculator (ASECalculator): The ASE calculator object.
        ase_optimizer (Optimizer): The ASE optimizer object.
        ase_optimizer_kwargs (dict): Keyword arguments for the ASE optimizer.

    Returns:
        Atoms: The optimized structure.
    """
    structure_optimized = structure.copy()
    structure_optimized.calc = ase_calculator
    ase_optimizer_obj = ase_optimizer(structure_optimized)
    ase_optimizer_obj.run(**ase_optimizer_kwargs)
    return structure_optimized


def optimize_positions_and_volume_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    ase_optimizer: type[Optimizer],
    ase_optimizer_kwargs: dict,
    filter_class: type[Filter] = UnitCellFilter,
) -> Atoms:
    """
    Optimize the atomic positions and cell volume of the structure using ASE optimizer.

    Args:
        structure (Atoms): The ASE structure object.
        ase_calculator (ASECalculator): The ASE calculator object.
        ase_optimizer (Optimizer): The ASE optimizer object.
        ase_optimizer_kwargs (dict): Keyword arguments for the ASE optimizer.
        filter_class (Filter): The ASE filter class to use for filtering during structure optimization.

    Returns:
        Atoms: The optimized structure.
    """
    structure_optimized = structure.copy()
    structure_optimized.calc = ase_calculator
    ase_optimizer_obj = ase_optimizer(filter_class(structure_optimized))
    ase_optimizer_obj.run(**ase_optimizer_kwargs)
    return structure_optimized


def optimize_volume_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    ase_optimizer: type[Optimizer],
    ase_optimizer_kwargs: dict,
    filter_class: type[Filter] = UnitCellFilter,
    hydrostatic_strain: bool = True,
) -> Atoms:
    """
    Optimize the cell volume of the structure using ASE optimizer.

    Args:
        structure (Atoms): The ASE structure object.
        ase_calculator (ASECalculator): The ASE calculator object.
        ase_optimizer (Optimizer): The ASE optimizer object.
        ase_optimizer_kwargs (dict): Keyword arguments for the ASE optimizer.
        filter_class (Filter): The ASE filter class to use for filtering during structure optimization.
        hydrostatic_strain (bool): Constrain the cell by only allowing hydrostatic deformation.

    Returns:
        Atoms: The optimized structure.
    """
    structure_optimized = structure.copy()
    structure_optimized.calc = ase_calculator
    structure_optimized.set_constraint(
        FixAtoms(np.ones(len(structure_optimized), dtype=bool))
    )
    ase_optimizer_obj = ase_optimizer(
        filter_class(
            atoms=structure_optimized,
            hydrostatic_strain=hydrostatic_strain,
        )
    )
    ase_optimizer_obj.run(**ase_optimizer_kwargs)
    return structure_optimized


def calc_molecular_dynamics_thermal_expansion_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    temperature_start: float = 15.0,
    temperature_stop: float = 1500.0,
    temperature_step: float = 5.0,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 1 * units.fs,
    ttime: float = 100 * units.fs,
    pfactor: float = 2e6 * units.GPa * (units.fs**2),
    externalstress: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar,
    output_keys: list[str] = OutputThermalExpansion.keys(),
) -> dict:
    """
    Calculate thermal expansion using molecular dynamics simulation with ASE.

    Args:
        structure (Atoms): The atomic structure to simulate.
        ase_calculator (ASECalculator): The ASE calculator to use for energy and force calculations.
        temperature_start (float, optional): The starting temperature in Kelvin. Defaults to 15.0.
        temperature_stop (float, optional): The stopping temperature in Kelvin. Defaults to 1500.0.
        temperature_step (float, optional): The temperature step size in Kelvin. Defaults to 5.0.
        run (int, optional): The number of MD steps to perform. Defaults to 100.
        thermo (int, optional): The interval at which to print thermodynamic properties. Defaults to 100.
        timestep (float, optional): The time step size in fs. Defaults to 1 * units.fs.
        ttime (float, optional): The total time for the simulation in fs. Defaults to 100 * units.fs.
        pfactor (float, optional): The pressure factor in GPa * fs^2. Defaults to 2e6 * units.GPa * (units.fs**2).
        externalstress (np.ndarray, optional): The external stress tensor in bar. Defaults to np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar.
        output_keys (list[str], optional): The keys of the properties to include in the output dictionary. Defaults to OutputThermalExpansion.keys().

    Returns:
        dict: A dictionary containing the calculated thermal expansion properties.
    """
    structure_current = structure.copy()
    temperature_lst = np.arange(
        temperature_start, temperature_stop + temperature_step, temperature_step
    ).tolist()
    volume_md_lst, temperature_md_lst = [], []
    for temperature in get_tqdm_iterator(temperature_lst):
        result_dict = calc_molecular_dynamics_npt_with_ase(
            structure=structure_current.copy(),
            ase_calculator=ase_calculator,
            run=run,
            thermo=thermo,
            timestep=timestep,
            ttime=ttime,
            pfactor=pfactor,
            temperature=temperature,
            externalstress=externalstress,
        )
        structure_current.set_cell(cell=result_dict["cell"][-1], scale_atoms=True)
        temperature_md_lst.append(result_dict["temperature"][-1])
        volume_md_lst.append(result_dict["volume"][-1])
    return get_thermal_expansion_output(
        temperatures_lst=temperature_md_lst,
        volumes_lst=volume_md_lst,
        output_keys=output_keys,
    )


def calc_molecular_dynamics_npt_with_ase(
    structure: Atoms,
    ase_calculator: ASECalculator,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 1 * units.fs,
    ttime: float = 100 * units.fs,
    pfactor: float = 2e6 * units.GPa * (units.fs**2),
    temperature: float = 300.0,
    externalstress: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar,
    output_keys: list[str] = OutputMolecularDynamics.keys(),
) -> dict:
    """
    Perform NPT molecular dynamics simulation using ASE.

    Args:
        structure (Atoms): The atomic structure to simulate.
        ase_calculator (ASECalculator): The ASE calculator to use for energy and force calculations.
        run (int, optional): The number of MD steps to perform. Defaults to 100.
        thermo (int, optional): The interval at which to print thermodynamic properties. Defaults to 100.
        timestep (float, optional): The time step size in fs. Defaults to 1 * units.fs.
        ttime (float, optional): The total time for the simulation in fs. Defaults to 100 * units.fs.
        pfactor (float, optional): The pressure factor in GPa * fs^2. Defaults to 2e6 * units.GPa * (units.fs**2).
        temperature (float, optional): The desired temperature in Kelvin. Defaults to 300.0.
        externalstress (np.ndarray, optional): The external stress tensor in bar. Defaults to np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * units.bar.
        output_keys (list[str], optional): The keys of the properties to include in the output dictionary. Defaults to OutputMolecularDynamics.keys().

    Returns:
        dict: A dictionary containing the calculated properties at each MD step.
    """
    return _calc_molecular_dynamics_with_ase(
        dyn=NPT(
            atoms=structure,
            timestep=timestep,
            temperature=None,
            externalstress=externalstress,
            ttime=ttime,
            pfactor=pfactor,
            temperature_K=temperature,
            mask=None,
            trajectory=None,
            logfile=None,
            loginterval=1,
            append_trajectory=False,
        ),
        structure=structure,
        ase_calculator=ase_calculator,
        temperature=temperature,
        run=run,
        thermo=thermo,
        output_keys=output_keys,
    )


def _calc_molecular_dynamics_with_ase(
    dyn,
    structure: Atoms,
    ase_calculator: ASECalculator,
    temperature: float,
    run: int,
    thermo: int,
    output_keys: list[str],
) -> dict:
    """
    Perform molecular dynamics simulation using ASE.

    Args:
        dyn (ase.md.MDLogger): The ASE dynamics object.
        structure (Atoms): The atomic structure to simulate.
        ase_calculator (ASECalculator): The ASE calculator to use for energy and force calculations.
        temperature (float): The desired temperature in Kelvin.
        run (int): The number of MD steps to perform.
        thermo (int): The interval at which to print thermodynamic properties.
        output_keys (list[str]): The keys of the properties to include in the output dictionary.

    Returns:
        dict: A dictionary containing the calculated properties at each MD step.
    """
    structure.calc = ase_calculator
    MaxwellBoltzmannDistribution(atoms=structure, temperature_K=temperature)
    cache = {q: [] for q in output_keys}
    for _i in range(int(run / thermo)):
        dyn.run(thermo)
        ase_instance = ASEExecutor(
            ase_structure=structure, ase_calculator=ase_calculator
        )
        calc_dict = OutputMolecularDynamics(
            **{k: getattr(ase_instance, k) for k in OutputMolecularDynamics.keys()}
        ).get(output_keys=output_keys)
        for k, v in calc_dict.items():
            cache[k].append(v)
    return {q: np.array(cache[q]) for q in output_keys}
