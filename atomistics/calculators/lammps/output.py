import dataclasses

from pylammpsmpi import LammpsASELibrary


@dataclasses.dataclass
class LammpsOutput:
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    @classmethod
    def get(cls, engine: LammpsASELibrary, *quantities: str) -> dict:
        return {q: getattr(cls, q)(engine) for q in quantities}


@dataclasses.dataclass
class LammpsMDOutput(LammpsOutput):
    positions: callable = LammpsASELibrary.interactive_positions_getter
    cell: callable = LammpsASELibrary.interactive_cells_getter
    forces: callable = LammpsASELibrary.interactive_forces_getter
    temperature: callable = LammpsASELibrary.interactive_temperatures_getter
    energy_pot: callable = LammpsASELibrary.interactive_energy_pot_getter
    energy_tot: callable = LammpsASELibrary.interactive_energy_tot_getter
    pressure: callable = LammpsASELibrary.interactive_pressures_getter
    velocities: callable = LammpsASELibrary.interactive_velocities_getter


@dataclasses.dataclass
class LammpsStaticOutput(LammpsOutput):
    forces: callable = LammpsASELibrary.interactive_forces_getter
    energy: callable = LammpsASELibrary.interactive_energy_pot_getter
    stress: callable = LammpsASELibrary.interactive_pressures_getter
