import dataclasses


@dataclasses.dataclass
class EngineOutput:
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    def get(self, engine, *output: str) -> dict:
        return {q: getattr(self, q)(engine) for q in output}


@dataclasses.dataclass
class OutputStatic(EngineOutput):
    forces: callable
    energy: callable
    stress: callable
    volume: callable


@dataclasses.dataclass
class OutputMolecularDynamics(EngineOutput):
    positions: callable
    cell: callable
    forces: callable
    temperature: callable
    energy_pot: callable
    energy_tot: callable
    pressure: callable
    velocities: callable
    volume: callable


@dataclasses.dataclass
class OutputThermalExpansion(EngineOutput):
    temperatures: callable
    volumes: callable


@dataclasses.dataclass
class OutputThermodynamic(OutputThermalExpansion):
    free_energy: callable
    entropy: callable
    heat_capacity: callable


@dataclasses.dataclass
class EquilibriumEnergy(EngineOutput):
    energy_eq: callable


@dataclasses.dataclass
class EquilibriumVolume(EngineOutput):
    volume_eq: callable


@dataclasses.dataclass
class EquilibriumBulkModul(EngineOutput):
    bulkmodul_eq: callable


@dataclasses.dataclass
class EquilibriumBulkModulDerivative(EngineOutput):
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
class OutputPhonons(EngineOutput):
    mesh_dict: callable
    band_structure_dict: callable
    total_dos_dict: callable
    dynamical_matrix: callable
    force_constants: callable
