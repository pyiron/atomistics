from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas
from jinja2 import Template
from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.lammps.commands import (
    LAMMPS_ENSEMBLE_NPH,
    LAMMPS_ENSEMBLE_NPT,
    LAMMPS_ENSEMBLE_NVT,
    LAMMPS_LANGEVIN,
    LAMMPS_MINIMIZE,
    LAMMPS_NVE,
    LAMMPS_RUN,
    LAMMPS_THERMO,
    LAMMPS_THERMO_STYLE,
    LAMMPS_TIMESTEP,
    LAMMPS_VELOCITY,
)
from atomistics.calculators.lammps.helpers import (
    lammps_calc_md,
    lammps_run,
    lammps_shutdown,
    lammps_thermal_expansion_loop,
)
from atomistics.calculators.lammps.shared import get_box_relax_command
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputMolecularDynamics, OutputStatic
from atomistics.shared.thermal_expansion import OutputThermalExpansion

if TYPE_CHECKING:
    from ase import Atoms
    from pandas import DataFrame
    from pylammpsmpi import LammpsASELibrary

    from atomistics.calculators.interface import TaskName


def optimize_positions_and_volume_with_lammpslib(
    structure: Atoms,
    potential_dataframe: DataFrame,
    min_style: str = "cg",
    etol: float = 0.0,
    ftol: float = 0.0001,
    maxiter: int = 100000,
    maxeval: int = 10000000,
    thermo: int = 10,
    pressure: float | Iterable[float | None] = 0.0,
    vmax: float | None = None,
    lmp=None,
    **kwargs,
) -> Atoms:
    """
    Relax atomic positions and cell using the LAMMPS library interface.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        min_style (str): LAMMPS minimisation style. Defaults to ``"cg"``.
        etol (float): Energy tolerance for minimisation. Defaults to ``0.0``.
        ftol (float): Force tolerance in eV/Å. Defaults to ``0.0001``.
        maxiter (int): Maximum number of minimisation iterations. Defaults to ``100000``.
        maxeval (int): Maximum number of force evaluations. Defaults to ``10000000``.
        thermo (int): Thermo output frequency. Defaults to ``10``.
        pressure (float | Iterable[float | None]): Target pressure for ``box/relax`` in bar.
        vmax (float | None): Maximum fractional volume change per step for ``box/relax``.
        lmp: Existing LAMMPS library instance to reuse. A new instance is created if ``None``.
        **kwargs: Additional keyword arguments forwarded to ``lammps_run``.

    Returns:
        Atoms: A copy of the input structure with relaxed positions and cell.
    """
    template_str = "\n".join(
        [
            get_box_relax_command(pressure=pressure, vmax=vmax),
            LAMMPS_THERMO_STYLE,
            LAMMPS_THERMO,
            LAMMPS_MINIMIZE,
        ]
    )
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(template_str).render(
            min_style=min_style,
            etol=etol,
            ftol=ftol,
            maxiter=maxiter,
            maxeval=maxeval,
            thermo=thermo,
        ),
        lmp=lmp,
        **kwargs,
    )
    structure_copy = structure.copy()
    structure_copy.set_cell(lmp_instance.interactive_cells_getter(), scale_atoms=True)
    structure_copy.positions = lmp_instance.interactive_positions_getter()
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return structure_copy


def optimize_positions_with_lammpslib(
    structure: Atoms,
    potential_dataframe: DataFrame,
    min_style: str = "cg",
    etol: float = 0.0,
    ftol: float = 0.0001,
    maxiter: int = 100000,
    maxeval: int = 10000000,
    thermo: int = 10,
    lmp=None,
    **kwargs,
) -> Atoms:
    """
    Relax atomic positions using the LAMMPS library interface (cell fixed).

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        min_style (str): LAMMPS minimisation style. Defaults to ``"cg"``.
        etol (float): Energy tolerance for minimisation. Defaults to ``0.0``.
        ftol (float): Force tolerance in eV/Å. Defaults to ``0.0001``.
        maxiter (int): Maximum number of minimisation iterations. Defaults to ``100000``.
        maxeval (int): Maximum number of force evaluations. Defaults to ``10000000``.
        thermo (int): Thermo output frequency. Defaults to ``10``.
        lmp: Existing LAMMPS library instance to reuse. A new instance is created if ``None``.
        **kwargs: Additional keyword arguments forwarded to ``lammps_run``.

    Returns:
        Atoms: A copy of the input structure with relaxed atomic positions.
    """
    template_str = "\n".join([LAMMPS_THERMO_STYLE, LAMMPS_THERMO, LAMMPS_MINIMIZE])
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(template_str).render(
            min_style=min_style,
            etol=etol,
            ftol=ftol,
            maxiter=maxiter,
            maxeval=maxeval,
            thermo=thermo,
        ),
        lmp=lmp,
        **kwargs,
    )
    structure_copy = structure.copy()
    structure_copy.positions = lmp_instance.interactive_positions_getter()
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return structure_copy


def calc_static_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    lmp=None,
    output_keys=OutputStatic.keys(),
    **kwargs,
) -> dict:
    """
    Run a static calculation using the LAMMPS library interface.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        lmp: Existing LAMMPS library instance to reuse. A new instance is created if ``None``.
        output_keys: Which output quantities to return. Defaults to all ``OutputStatic`` keys.
        **kwargs: Additional keyword arguments forwarded to ``lammps_run``.

    Returns:
        dict: Requested output quantities keyed by name.
    """
    template_str = "\n".join([LAMMPS_THERMO_STYLE, LAMMPS_THERMO, LAMMPS_RUN])
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(template_str).render(
            run=0,
            thermo=100,
        ),
        lmp=lmp,
        **kwargs,
    )
    result_dict = OutputStatic(
        forces=lmp_instance.interactive_forces_getter,
        energy=lmp_instance.interactive_energy_pot_getter,
        stress=lmp_instance.interactive_pressures_getter,
        volume=lmp_instance.interactive_volume_getter,
    ).get(output_keys=output_keys)
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_nvt_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    Tstart: float = 100.0,
    Tstop: float = 100.0,
    Tdamp: float = 0.1,
    run: int = 100,
    thermo: int = 10,
    timestep: float = 0.001,
    seed: int = 4928459,
    dist: str = "gaussian",
    velocity_rescale_factor: float | None = 2.0,
    lmp=None,
    output_keys=OutputMolecularDynamics.keys(),
    **kwargs,
) -> dict:
    """
    Run NVT (canonical ensemble) molecular dynamics using the LAMMPS library interface.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        Tstart (float): Starting temperature in K. Defaults to ``100.0``.
        Tstop (float): Target temperature in K at the end of the run. Defaults to ``100.0``.
        Tdamp (float): Nosé-Hoover thermostat damping parameter in ps. Defaults to ``0.1``.
        run (int): Total number of MD timesteps. Defaults to ``100``.
        thermo (int): Number of timesteps between output snapshots. Defaults to ``10``.
        timestep (float): MD timestep in ps. Defaults to ``0.001``.
        seed (int): Random seed for velocity initialisation. Defaults to ``4928459``.
        dist (str): Velocity distribution type (``"gaussian"`` or ``"uniform"``). Defaults to ``"gaussian"``.
        velocity_rescale_factor (float | None): Scaling factor for initial velocity rescaling.
            No rescaling is applied when ``None``. Defaults to ``2.0``.
        lmp: Existing LAMMPS library instance to reuse. A new instance is created if ``None``.
        output_keys: Which output quantities to return. Defaults to all ``OutputMolecularDynamics`` keys.
        **kwargs: Additional keyword arguments forwarded to ``lammps_run``.

    Returns:
        dict: Output quantities as numpy arrays with one entry per snapshot, keyed by quantity name.
    """
    if velocity_rescale_factor is not None:
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_VELOCITY,
                LAMMPS_ENSEMBLE_NVT,
            ]
        )
        input_template = Template(init_str).render(
            thermo=thermo,
            Tstart=Tstart,
            temp=Tstart,
            Tstop=Tstop,
            Tdamp=Tdamp,
            timestep=timestep,
            seed=seed,
            dist=dist,
            velocity_rescale_factor=velocity_rescale_factor,
        )
    else:
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_ENSEMBLE_NVT,
            ]
        )
        input_template = Template(init_str).render(
            thermo=thermo,
            Tstart=Tstart,
            temp=Tstart,
            Tstop=Tstop,
            Tdamp=Tdamp,
            timestep=timestep,
        )
    run_str = LAMMPS_RUN + "\n"
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=input_template,
        lmp=lmp,
        **kwargs,
    )
    result_dict = lammps_calc_md(
        lmp_instance=lmp_instance,
        run_str=run_str,
        run=run,
        thermo=thermo,
        output_keys=output_keys,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_npt_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    Tstart: float = 100.0,
    Tstop: float = 100.0,
    Tdamp: float = 0.1,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Pstart: float = 0.0,
    Pstop: float = 0.0,
    Pdamp: float = 1.0,
    seed: int = 4928459,
    dist: str = "gaussian",
    couple_xyz: bool = False,
    velocity_rescale_factor: float | None = 2.0,
    lmp=None,
    output_keys=OutputMolecularDynamics.keys(),
    **kwargs,
) -> dict:
    """
    Run NPT (isothermal-isobaric ensemble) molecular dynamics using the LAMMPS library interface.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        Tstart (float): Starting temperature in K. Defaults to ``100.0``.
        Tstop (float): Target temperature in K at the end of the run. Defaults to ``100.0``.
        Tdamp (float): Thermostat damping parameter in ps. Defaults to ``0.1``.
        run (int): Total number of MD timesteps. Defaults to ``100``.
        thermo (int): Number of timesteps between output snapshots. Defaults to ``100``.
        timestep (float): MD timestep in ps. Defaults to ``0.001``.
        Pstart (float): Starting pressure in bar. Defaults to ``0.0``.
        Pstop (float): Target pressure in bar at the end of the run. Defaults to ``0.0``.
        Pdamp (float): Barostat damping parameter in ps. Defaults to ``1.0``.
        seed (int): Random seed for velocity initialisation. Defaults to ``4928459``.
        dist (str): Velocity distribution type. Defaults to ``"gaussian"``.
        couple_xyz (bool): Whether to couple all three box dimensions (isotropic pressure).
            Defaults to ``False``.
        velocity_rescale_factor (float | None): Scaling factor for initial velocity rescaling.
            No rescaling is applied when ``None``. Defaults to ``2.0``.
        lmp: Existing LAMMPS library instance to reuse. A new instance is created if ``None``.
        output_keys: Which output quantities to return. Defaults to all ``OutputMolecularDynamics`` keys.
        **kwargs: Additional keyword arguments forwarded to ``lammps_run``.

    Returns:
        dict: Output quantities as numpy arrays with one entry per snapshot, keyed by quantity name.
    """
    if couple_xyz:
        LAMMPS_ENSEMBLE_NPT_XYZ = LAMMPS_ENSEMBLE_NPT + " couple xyz"
    else:
        LAMMPS_ENSEMBLE_NPT_XYZ = LAMMPS_ENSEMBLE_NPT
    if velocity_rescale_factor is not None:
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_VELOCITY,
                LAMMPS_ENSEMBLE_NPT_XYZ,
            ]
        )
        input_template = Template(init_str).render(
            thermo=thermo,
            Tstart=Tstart,
            temp=Tstart,
            Tstop=Tstop,
            Tdamp=Tdamp,
            Pstart=Pstart,
            Pstop=Pstop,
            Pdamp=Pdamp,
            timestep=timestep,
            seed=seed,
            dist=dist,
            velocity_rescale_factor=velocity_rescale_factor,
        )
    else:
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_ENSEMBLE_NPT_XYZ,
            ]
        )
        input_template = Template(init_str).render(
            thermo=thermo,
            Tstart=Tstart,
            temp=Tstart,
            Tstop=Tstop,
            Tdamp=Tdamp,
            Pstart=Pstart,
            Pstop=Pstop,
            Pdamp=Pdamp,
            timestep=timestep,
        )
    run_str = LAMMPS_RUN + "\n"
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=input_template,
        lmp=lmp,
        **kwargs,
    )
    result_dict = lammps_calc_md(
        lmp_instance=lmp_instance,
        run_str=run_str,
        run=run,
        thermo=thermo,
        output_keys=output_keys,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_nph_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Tstart: float = 100.0,
    Pstart: float = 0.0,
    Pstop: float = 0.0,
    Pdamp: float = 1.0,
    seed: int = 4928459,
    dist: str = "gaussian",
    velocity_rescale_factor: float | None = 2.0,
    lmp=None,
    output_keys=OutputMolecularDynamics.keys(),
    **kwargs,
) -> dict:
    """
    Run NPH (isoenthalpic-isobaric ensemble) molecular dynamics using the LAMMPS library interface.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        run (int): Total number of MD timesteps. Defaults to ``100``.
        thermo (int): Number of timesteps between output snapshots. Defaults to ``100``.
        timestep (float): MD timestep in ps. Defaults to ``0.001``.
        Tstart (float): Initial temperature in K used for velocity initialisation. Defaults to ``100.0``.
        Pstart (float): Starting pressure in bar. Defaults to ``0.0``.
        Pstop (float): Target pressure in bar at the end of the run. Defaults to ``0.0``.
        Pdamp (float): Barostat damping parameter in ps. Defaults to ``1.0``.
        seed (int): Random seed for velocity initialisation. Defaults to ``4928459``.
        dist (str): Velocity distribution type. Defaults to ``"gaussian"``.
        velocity_rescale_factor (float | None): Scaling factor for initial velocity rescaling.
            No rescaling is applied when ``None``. Defaults to ``2.0``.
        lmp: Existing LAMMPS library instance to reuse. A new instance is created if ``None``.
        output_keys: Which output quantities to return. Defaults to all ``OutputMolecularDynamics`` keys.
        **kwargs: Additional keyword arguments forwarded to ``lammps_run``.

    Returns:
        dict: Output quantities as numpy arrays with one entry per snapshot, keyed by quantity name.
    """
    if velocity_rescale_factor is not None:
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_VELOCITY,
                LAMMPS_ENSEMBLE_NPH,
            ]
        )
        input_template = Template(init_str).render(
            thermo=thermo,
            temp=Tstart,
            Pstart=Pstart,
            Pstop=Pstop,
            Pdamp=Pdamp,
            timestep=timestep,
            seed=seed,
            dist=dist,
            velocity_rescale_factor=velocity_rescale_factor,
        )
    else:
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_ENSEMBLE_NPH,
            ]
        )
        input_template = Template(init_str).render(
            thermo=thermo,
            temp=Tstart,
            Pstart=Pstart,
            Pstop=Pstop,
            Pdamp=Pdamp,
            timestep=timestep,
        )
    run_str = LAMMPS_RUN + "\n"
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=input_template,
        lmp=lmp,
        **kwargs,
    )
    result_dict = lammps_calc_md(
        lmp_instance=lmp_instance,
        run_str=run_str,
        run=run,
        thermo=thermo,
        output_keys=output_keys,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_langevin_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Tstart: float = 100.0,
    Tstop: float = 100,
    Tdamp: float = 0.1,
    seed: int = 4928459,
    dist: str = "gaussian",
    velocity_rescale_factor: float | None = 2.0,
    lmp=None,
    output_keys=OutputMolecularDynamics.keys(),
    **kwargs,
) -> dict:
    """
    Run Langevin (NVE + Langevin thermostat) molecular dynamics using the LAMMPS library interface.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        run (int): Total number of MD timesteps. Defaults to ``100``.
        thermo (int): Number of timesteps between output snapshots. Defaults to ``100``.
        timestep (float): MD timestep in ps. Defaults to ``0.001``.
        Tstart (float): Starting temperature for the Langevin thermostat in K. Defaults to ``100.0``.
        Tstop (float): Target temperature for the Langevin thermostat in K. Defaults to ``100``.
        Tdamp (float): Langevin thermostat damping parameter in ps. Defaults to ``0.1``.
        seed (int): Random seed for velocity initialisation and Langevin noise. Defaults to ``4928459``.
        dist (str): Velocity distribution type. Defaults to ``"gaussian"``.
        velocity_rescale_factor (float | None): Scaling factor for initial velocity rescaling.
            No rescaling is applied when ``None``. Defaults to ``2.0``.
        lmp: Existing LAMMPS library instance to reuse. A new instance is created if ``None``.
        output_keys: Which output quantities to return. Defaults to all ``OutputMolecularDynamics`` keys.
        **kwargs: Additional keyword arguments forwarded to ``lammps_run``.

    Returns:
        dict: Output quantities as numpy arrays with one entry per snapshot, keyed by quantity name.
    """
    if velocity_rescale_factor is not None:
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_VELOCITY,
                LAMMPS_NVE,
                LAMMPS_LANGEVIN,
            ]
        )
        input_template = Template(init_str).render(
            thermo=thermo,
            temp=Tstart,
            Tstart=Tstart,
            Tstop=Tstop,
            Tdamp=Tdamp,
            timestep=timestep,
            seed=seed,
            dist=dist,
            velocity_rescale_factor=velocity_rescale_factor,
        )
    else:
        init_str = "\n".join(
            [
                LAMMPS_THERMO_STYLE,
                LAMMPS_TIMESTEP,
                LAMMPS_THERMO,
                LAMMPS_NVE,
                LAMMPS_LANGEVIN,
            ]
        )
        input_template = Template(init_str).render(
            thermo=thermo,
            temp=Tstart,
            Tstart=Tstart,
            Tstop=Tstop,
            Tdamp=Tdamp,
            seed=seed,
            timestep=timestep,
        )
    run_str = LAMMPS_RUN + "\n"
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=input_template,
        lmp=lmp,
        **kwargs,
    )
    result_dict = lammps_calc_md(
        lmp_instance=lmp_instance,
        run_str=run_str,
        run=run,
        thermo=thermo,
        output_keys=output_keys,
    )
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return result_dict


def calc_molecular_dynamics_thermal_expansion_with_lammpslib(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    Tstart: float = 15.0,
    Tstop: float = 1500.0,
    Tstep: int = 5,
    Tdamp: float = 0.1,
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Pstart: float = 0.0,
    Pstop: float = 0.0,
    Pdamp: float = 1.0,
    seed: int = 4928459,
    dist: str = "gaussian",
    couple_xyz: bool = False,
    lmp: LammpsASELibrary | None = None,
    output_keys: Iterable[str] = OutputThermalExpansion.keys(),
    **kwargs,
) -> dict:
    """
    Compute thermal expansion via NPT MD across a temperature range using the LAMMPS library interface.

    Runs ``lammps_thermal_expansion_loop`` over temperatures from ``Tstart`` to ``Tstop``
    in steps of ``Tstep``.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        Tstart (float): Starting temperature in K. Defaults to ``15.0``.
        Tstop (float): Ending temperature in K (inclusive). Defaults to ``1500.0``.
        Tstep (int): Temperature increment in K. Defaults to ``5``.
        Tdamp (float): Thermostat damping parameter in ps. Defaults to ``0.1``.
        run (int): Number of MD timesteps per temperature point. Defaults to ``100``.
        thermo (int): Thermo output frequency in timesteps. Defaults to ``100``.
        timestep (float): MD timestep in ps. Defaults to ``0.001``.
        Pstart (float): Starting pressure in bar. Defaults to ``0.0``.
        Pstop (float): Ending pressure in bar. Defaults to ``0.0``.
        Pdamp (float): Barostat damping parameter in ps. Defaults to ``1.0``.
        seed (int): Random seed for velocity initialisation. Defaults to ``4928459``.
        dist (str): Velocity distribution type. Defaults to ``"gaussian"``.
        couple_xyz (bool): Whether to couple all three box dimensions (isotropic pressure).
            Defaults to ``False``.
        lmp (LammpsASELibrary | None): Existing LAMMPS library instance to reuse.
        output_keys (Iterable[str]): Which output quantities to return.
        **kwargs: Additional keyword arguments forwarded to ``lammps_thermal_expansion_loop``.

    Returns:
        dict: Thermal expansion output (temperatures and volumes) keyed by quantity name.
    """
    init_str = "\n".join(
        [
            LAMMPS_THERMO_STYLE,
            LAMMPS_TIMESTEP,
            LAMMPS_THERMO,
            LAMMPS_VELOCITY,
            "",
        ]
    )
    if couple_xyz:
        LAMMPS_ENSEMBLE_NPT_XYZ = LAMMPS_ENSEMBLE_NPT + " couple xyz"
    else:
        LAMMPS_ENSEMBLE_NPT_XYZ = LAMMPS_ENSEMBLE_NPT
    run_str = "\n".join([LAMMPS_ENSEMBLE_NPT_XYZ, LAMMPS_RUN])
    temperature_lst = np.arange(Tstart, Tstop + Tstep, Tstep).tolist()
    return lammps_thermal_expansion_loop(
        structure=structure,
        potential_dataframe=potential_dataframe,
        init_str=init_str,
        run_str=run_str,
        temperature_lst=temperature_lst,
        run=run,
        thermo=thermo,
        timestep=timestep,
        Tdamp=Tdamp,
        Pstart=Pstart,
        Pstop=Pstop,
        Pdamp=Pdamp,
        seed=seed,
        dist=dist,
        lmp=lmp,
        output_keys=output_keys,
        **kwargs,
    )


@as_task_dict_evaluator
def evaluate_with_lammpslib_library_interface(
    structure: Atoms,
    tasks: list[TaskName],
    potential_dataframe: DataFrame,
    lmp: LammpsASELibrary,
    lmp_optimizer_kwargs: dict | None = None,
) -> dict:
    """
    Evaluate a single structure for a list of tasks using an existing LAMMPS library instance.

    Decorated with ``as_task_dict_evaluator`` to handle task dict conversion. Used internally
    by ``evaluate_with_lammpslib`` which manages the LAMMPS instance lifecycle.

    Args:
        structure (Atoms): The input structure.
        tasks (list[TaskName]): List of task names to evaluate.
        potential_dataframe (DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        lmp (LammpsASELibrary): An active LAMMPS library instance.
        lmp_optimizer_kwargs (dict | None): Extra keyword arguments forwarded to the
            underlying calculation functions.

    Returns:
        dict: Results keyed by output quantity name.

    Raises:
        ValueError: If none of the requested tasks are implemented by this calculator.
    """
    if lmp_optimizer_kwargs is None:
        lmp_optimizer_kwargs = {}
    results: dict[str, Any] = {}
    if "optimize_positions_and_volume" in tasks:
        results["structure_with_optimized_positions_and_volume"] = (
            optimize_positions_and_volume_with_lammpslib(
                structure=structure,
                potential_dataframe=potential_dataframe,
                lmp=lmp,
                **lmp_optimizer_kwargs,
            )
        )
    elif "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = (
            optimize_positions_with_lammpslib(
                structure=structure,
                potential_dataframe=potential_dataframe,
                lmp=lmp,
                **lmp_optimizer_kwargs,
            )
        )
    elif "calc_molecular_dynamics_thermal_expansion" in tasks:
        results_dict = calc_molecular_dynamics_thermal_expansion_with_lammpslib(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            **lmp_optimizer_kwargs,
        )
        results["volume_over_temperature"] = (
            results_dict["temperatures"],
            results_dict["volumes"],
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_lammpslib(
            structure=structure,
            potential_dataframe=potential_dataframe,
            lmp=lmp,
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The LAMMPS calculator does not implement:", tasks)
    return results


def evaluate_with_lammpslib(
    task_dict: dict[str, dict[str, Atoms]],
    potential_dataframe: DataFrame,
    working_directory: str | None = None,
    cores: int = 1,
    comm: object | None = None,
    logger: object | None = None,
    log_file: str | None = None,
    library: object | None = None,
    disable_log_file: bool = True,
    lmp_optimizer_kwargs: dict | None = None,
) -> dict:
    """
    Evaluate a task dictionary using the LAMMPS library interface, managing the instance lifecycle.

    Creates a ``LammpsASELibrary`` instance, delegates to
    ``evaluate_with_lammpslib_library_interface``, then closes the instance.

    Args:
        task_dict (dict[str, dict[str, Atoms]]): Task dictionary mapping task names to structure dicts.
        potential_dataframe (DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        working_directory (str | None): Working directory for LAMMPS. Defaults to ``None``.
        cores (int): Number of MPI cores to use. Defaults to ``1``.
        comm: MPI communicator object. Defaults to ``None``.
        logger: Logger object. Defaults to ``None``.
        log_file (str | None): Path to the LAMMPS log file. Defaults to ``None``.
        library: Pre-loaded LAMMPS shared library object. Defaults to ``None``.
        disable_log_file (bool): Whether to suppress the LAMMPS log file. Defaults to ``True``.
        lmp_optimizer_kwargs (dict | None): Extra keyword arguments forwarded to the
            underlying calculation functions.

    Returns:
        dict: Results keyed by output quantity name.
    """
    if lmp_optimizer_kwargs is None:
        lmp_optimizer_kwargs = {}
    lmp = LammpsASELibrary(
        working_directory=working_directory,
        cores=cores,
        comm=comm,
        logger=logger,
        log_file=log_file,
        library=library,
        disable_log_file=disable_log_file,
    )
    results_dict = evaluate_with_lammpslib_library_interface(
        task_dict=task_dict,
        potential_dataframe=potential_dataframe,
        lmp=lmp,
        lmp_optimizer_kwargs=lmp_optimizer_kwargs,
    )
    lmp.close()
    return results_dict
