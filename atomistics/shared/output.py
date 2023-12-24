import dataclasses


@dataclasses.dataclass
class Output:
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    def get(self, engine, *output: str) -> dict:
        return {q: getattr(self, q)(engine) for q in output}


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


@dataclasses.dataclass
class OutputThermodynamic(Output):
    temperatures: callable
    free_energy: callable
    entropy: callable
    heat_capacity: callable
    volumes: callable


@dataclasses.dataclass
class EquilibriumEnergy(Output):
    energy_eq: callable


@dataclasses.dataclass
class EquilibriumVolume(Output):
    volume_eq: callable


@dataclasses.dataclass
class EquilibriumBulkModul(Output):
    bulkmodul_eq: callable


@dataclasses.dataclass
class EquilibriumBulkModulDerivative(Output):
    b_prime_eq: callable


@dataclasses.dataclass
class OutputEnergyVolumeCurve(
    EquilibriumEnergy,
    EquilibriumVolume,
    EquilibriumBulkModul,
    EquilibriumBulkModulDerivative,
):
    fit_dict: callable
    energy: callable
    volume: callable


@dataclasses.dataclass
class OutputElastic(Output):
    C: callable
    S: callable
    BV: callable
    BR: callable
    BH: callable
    GV: callable
    GR: callable
    GH: callable
    EV: callable
    ER: callable
    EH: callable
    nuV: callable
    nuR: callable
    nuH: callable
    AVR: callable
    C_eigval: callable
