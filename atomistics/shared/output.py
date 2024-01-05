import dataclasses

from atomistics.shared.generic import (
    static_calculation_output_keys,
    molecular_dynamics_output_keys,
    thermal_expansion_output_keys,
    thermodynamic_output_keys,
    energy_volume_curve_output_keys,
    elastic_matrix_output_keys,
    phonon_output_keys,
)


@dataclasses.dataclass
class Output:
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    def get(self, engine, *output: str) -> dict:
        return {q: getattr(self, q)(engine) for q in output}


OutputStatic = dataclasses.make_dataclass(
    cls_name="OutputStatic",
    fields=[(key, callable) for key in static_calculation_output_keys],
    bases=(Output, )
)


OutputMolecularDynamics = dataclasses.make_dataclass(
    cls_name="OutputMolecularDynamics",
    fields=[(key, callable) for key in molecular_dynamics_output_keys],
    bases=(Output, )
)


OutputThermalExpansion = dataclasses.make_dataclass(
    cls_name="OutputThermalExpansion",
    fields=[(key, callable) for key in thermal_expansion_output_keys],
    bases=(Output, )
)


OutputThermodynamic = dataclasses.make_dataclass(
    cls_name="OutputThermodynamic",
    fields=[(key, callable) for key in thermodynamic_output_keys],
    bases=(Output, )
)


OutputEnergyVolumeCurve = dataclasses.make_dataclass(
    cls_name="OutputEnergyVolumeCurve",
    fields=[(key, callable) for key in energy_volume_curve_output_keys],
    bases=(Output, )
)


OutputElastic = dataclasses.make_dataclass(
    cls_name="OutputElastic",
    fields=[(key, callable) for key in elastic_matrix_output_keys],
    bases=(Output, )
)


OutputPhonons = dataclasses.make_dataclass(
    cls_name="OutputPhonons",
    fields=[(key, callable) for key in phonon_output_keys],
    bases=(Output, )
)
