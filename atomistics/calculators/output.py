import dataclasses


@dataclasses.dataclass
class Output:
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    def get(self, engine, *quantities: str) -> dict:
        return {q: getattr(self, q)(engine) for q in quantities}


@dataclasses.dataclass
class OutputStatic(Output):
    forces: callable
    energy: callable
    stress: callable


@dataclasses.dataclass
class OutputMolecularDynamics(Output):
    positions: callable
    cell: callable
    forces: callable
    temperature: callable
    energy_pot: callable
    energy_tot: callable
    pressure: callable
    velocities: callable
