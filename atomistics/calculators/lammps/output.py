import dataclasses

from pylammpsmpi import LammpsASELibrary

from atomistics.calculators.output import AtomisticsOutput


@dataclasses.dataclass
class LammpsMDOutput(AtomisticsOutput):
    positions: callable = LammpsASELibrary.interactive_positions_getter
    cell: callable = LammpsASELibrary.interactive_cells_getter
    forces: callable = LammpsASELibrary.interactive_forces_getter
    temperature: callable = LammpsASELibrary.interactive_temperatures_getter
    energy_pot: callable = LammpsASELibrary.interactive_energy_pot_getter
    energy_tot: callable = LammpsASELibrary.interactive_energy_tot_getter
    pressure: callable = LammpsASELibrary.interactive_pressures_getter
    velocities: callable = LammpsASELibrary.interactive_velocities_getter


@dataclasses.dataclass
class LammpsStaticOutput(AtomisticsOutput):
    forces: callable = LammpsASELibrary.interactive_forces_getter
    energy: callable = LammpsASELibrary.interactive_energy_pot_getter
    stress: callable = LammpsASELibrary.interactive_pressures_getter
