from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas
from ase.atoms import Atoms
from jinja2 import Template
from lammpsparser import validate_potential_dataframe
from pylammpsmpi import LammpsASELibrary

from atomistics.shared.output import OutputMolecularDynamics, OutputThermalExpansion
from atomistics.shared.thermal_expansion import get_thermal_expansion_output
from atomistics.shared.tqdm_iterator import get_tqdm_iterator


def lammps_run(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    input_template: str | None = None,
    lmp: LammpsASELibrary | None = None,
    **kwargs,
) -> LammpsASELibrary:
    """
    Initialise a LAMMPS instance, load the structure and potential, and run an input template.

    If no ``lmp`` instance is provided a new one is created using ``**kwargs``.
    The structure is written to LAMMPS via the interactive interface, the potential
    commands are applied, and each line of ``input_template`` is sent as a library command.

    Args:
        structure (Atoms): The ASE structure to load into LAMMPS.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns
            describing the interatomic potential.
        input_template (str | None): Multi-line string of LAMMPS commands to execute after loading
            the structure and potential. No commands are sent if ``None``.
        lmp (LammpsASELibrary | None): An existing LAMMPS library instance to reuse.
            A new instance is created if ``None``.
        **kwargs: Additional keyword arguments forwarded to ``LammpsASELibrary`` when creating
            a new instance.

    Returns:
        LammpsASELibrary: The LAMMPS library instance after executing all commands.
    """
    potential_dataframe = validate_potential_dataframe(
        potential_dataframe=potential_dataframe
    )
    if lmp is None:
        lmp = LammpsASELibrary(**kwargs)

    # write structure to LAMMPS
    lmp.interactive_structure_setter(
        structure=structure,
        units="metal",
        dimension=3,
        boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
        atom_style="atomic",
        el_eam_lst=potential_dataframe.Species,
        calc_md=False,
    )

    # execute calculation
    for c in potential_dataframe.Config:
        lmp.interactive_lib_command(c)

    if input_template is not None:
        for line in input_template.split("\n"):
            lmp.interactive_lib_command(line)

    return lmp


def lammps_calc_md_step(
    lmp_instance: LammpsASELibrary,
    run_str: str,
    run: int,
    output_keys: Iterable[str] = OutputMolecularDynamics.keys(),
) -> dict:
    """
    Execute a single MD run segment and collect output quantities.

    Renders ``run_str`` with the given ``run`` count, sends it to the LAMMPS instance,
    and queries the instance for all requested output quantities.

    Args:
        lmp_instance (LammpsASELibrary): An active LAMMPS library instance.
        run_str (str): Jinja2 template string for the LAMMPS ``run`` command (receives ``run``).
        run (int): Number of timesteps to advance in this segment.
        output_keys (Iterable[str]): Names of the output quantities to collect.

    Returns:
        dict: Collected output quantities for this MD segment keyed by quantity name.
    """
    run_str_rendered = Template(run_str).render(run=run)
    lmp_instance.interactive_lib_command(run_str_rendered)
    return OutputMolecularDynamics(
        positions=lmp_instance.interactive_positions_getter,
        cell=lmp_instance.interactive_cells_getter,
        forces=lmp_instance.interactive_forces_getter,
        temperature=lmp_instance.interactive_temperatures_getter,
        energy_pot=lmp_instance.interactive_energy_pot_getter,
        energy_tot=lmp_instance.interactive_energy_tot_getter,
        pressure=lmp_instance.interactive_pressures_getter,
        velocities=lmp_instance.interactive_velocities_getter,
        volume=lmp_instance.interactive_volume_getter,
    ).get(output_keys=output_keys)


def lammps_calc_md(
    lmp_instance: LammpsASELibrary,
    run_str: str,
    run: int,
    thermo: int,
    output_keys: Iterable[str] = OutputMolecularDynamics.keys(),
) -> dict:
    """
    Run a full MD simulation by collecting output every ``thermo`` steps.

    Calls ``lammps_calc_md_step`` ``run // thermo`` times, each advancing ``thermo``
    timesteps, and stacks the per-step results into numpy arrays.

    Args:
        lmp_instance (LammpsASELibrary): An active LAMMPS library instance.
        run_str (str): Jinja2 template string for the LAMMPS ``run`` command (receives ``run``).
        run (int): Total number of MD timesteps to execute.
        thermo (int): Number of timesteps between output snapshots.
        output_keys (Iterable[str]): Names of the output quantities to collect.

    Returns:
        dict: Output quantities as numpy arrays with one entry per snapshot, keyed by quantity name.
    """
    results_lst = [
        lammps_calc_md_step(
            lmp_instance=lmp_instance,
            run_str=run_str,
            run=thermo,
            output_keys=output_keys,
        )
        for _ in range(run // thermo)
    ]
    return {q: np.array([d[q] for d in results_lst]) for q in output_keys}


def lammps_thermal_expansion_loop(
    structure: Atoms,
    potential_dataframe: pandas.DataFrame,
    init_str: str,
    run_str: str,
    temperature_lst: list[float],
    run: int = 100,
    thermo: int = 100,
    timestep: float = 0.001,
    Tdamp: float = 0.1,
    Pstart: float = 0.0,
    Pstop: float = 0.0,
    Pdamp: float = 1.0,
    seed: int = 4928459,
    dist: str = "gaussian",
    velocity_rescale_factor: float = 2.0,
    lmp=None,
    output_keys=OutputThermalExpansion.keys(),
    **kwargs,
) -> dict:
    """
    Run NPT molecular dynamics at a sequence of temperatures to compute thermal expansion.

    Initialises a LAMMPS simulation once using ``init_str`` and then iterates through
    ``temperature_lst``, running ``run`` steps at each temperature via ``run_str``.
    The equilibrium volume and temperature are recorded after each segment.

    Args:
        structure (Atoms): The input structure.
        potential_dataframe (pandas.DataFrame): DataFrame with ``"Species"`` and ``"Config"`` columns.
        init_str (str): Jinja2 template for the LAMMPS initialisation commands (thermostat setup etc.).
        run_str (str): Jinja2 template for the per-temperature run commands.
        temperature_lst (list[float]): Ordered list of target temperatures in K.
        run (int): Number of MD timesteps per temperature point. Defaults to ``100``.
        thermo (int): Thermo output frequency in timesteps. Defaults to ``100``.
        timestep (float): MD timestep in ps. Defaults to ``0.001``.
        Tdamp (float): Thermostat damping parameter in ps. Defaults to ``0.1``.
        Pstart (float): Starting pressure in bar. Defaults to ``0.0``.
        Pstop (float): Ending pressure in bar. Defaults to ``0.0``.
        Pdamp (float): Barostat damping parameter in ps. Defaults to ``1.0``.
        seed (int): Random seed for velocity initialisation. Defaults to ``4928459``.
        dist (str): Velocity distribution type (``"gaussian"`` or ``"uniform"``). Defaults to ``"gaussian"``.
        velocity_rescale_factor (float): Scaling factor applied during velocity initialisation. Defaults to ``2.0``.
        lmp: Existing LAMMPS instance to reuse. A new instance is created if ``None``.
        output_keys: Output quantities to return. Defaults to all ``OutputThermalExpansion`` keys.
        **kwargs: Additional keyword arguments forwarded to ``lammps_run``.

    Returns:
        dict: Thermal expansion output quantities (temperatures and volumes) keyed by name.
    """
    lmp_instance = lammps_run(
        structure=structure,
        potential_dataframe=potential_dataframe,
        input_template=Template(init_str).render(
            thermo=thermo,
            temp=temperature_lst[0],
            timestep=timestep,
            seed=seed,
            dist=dist,
            velocity_rescale_factor=velocity_rescale_factor,
        ),
        lmp=lmp,
        **kwargs,
    )

    volume_md_lst, temperature_md_lst = [], []
    for temp in get_tqdm_iterator(temperature_lst):
        run_str_rendered = Template(run_str).render(
            run=run,
            Tstart=temp - 5,
            Tstop=temp,
            Tdamp=Tdamp,
            Pstart=Pstart,
            Pstop=Pstop,
            Pdamp=Pdamp,
        )
        for line in run_str_rendered.split("\n"):
            lmp_instance.interactive_lib_command(line)
        volume_md_lst.append(lmp_instance.interactive_volume_getter())
        temperature_md_lst.append(lmp_instance.interactive_temperatures_getter())
    lammps_shutdown(lmp_instance=lmp_instance, close_instance=lmp is None)
    return get_thermal_expansion_output(
        temperatures_lst=np.array(temperature_md_lst),
        volumes_lst=np.array(volume_md_lst),
        output_keys=output_keys,
    )


def lammps_shutdown(
    lmp_instance: LammpsASELibrary, close_instance: bool = True
) -> None:
    """
    Issue a ``clear`` command to the LAMMPS instance and optionally close it.

    Args:
        lmp_instance (LammpsASELibrary): The active LAMMPS library instance to shut down.
        close_instance (bool): Whether to call ``lmp_instance.close()`` after clearing.
            Set to ``False`` when the caller manages the instance lifetime. Defaults to ``True``.
    """
    lmp_instance.interactive_lib_command("clear")
    if close_instance:
        lmp_instance.close()
