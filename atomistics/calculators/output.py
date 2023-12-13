import dataclasses


@dataclasses.dataclass
class AtomisticsOutput:
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    @classmethod
    def get(cls, engine, *quantities: str) -> dict:
        return {q: getattr(cls, q)(engine) for q in quantities}


@dataclasses.dataclass
class OutputStatic(AtomisticsOutput):
    forces: callable
    energy: callable
    stress: callable


@dataclasses.dataclass
class OutputMolecularDynamics(AtomisticsOutput):
    positions: callable
    cell: callable
    forces: callable
    temperature: callable
    energy_pot: callable
    energy_tot: callable
    pressure: callable
    velocities: callable
