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
    def get(self, engine, *output: str) -> dict:
        return {q: getattr(self, q)(engine) for q in output}


def make_output_dataclass(cls_name, output_keys):
    return dataclasses.make_dataclass(
        cls_name=cls_name,
        fields=[(key, callable) for key in output_keys],
        bases=(Output,),
    )


OutputStatic = make_output_dataclass(
    cls_name="OutputStatic", output_keys=static_calculation_output_keys
)


OutputMolecularDynamics = make_output_dataclass(
    cls_name="OutputMolecularDynamics",
    output_keys=molecular_dynamics_output_keys,
)


OutputThermalExpansion = make_output_dataclass(
    cls_name="OutputThermalExpansion",
    output_keys=thermal_expansion_output_keys,
)


OutputThermodynamic = make_output_dataclass(
    cls_name="OutputThermodynamic",
    output_keys=thermodynamic_output_keys,
)


OutputEnergyVolumeCurve = make_output_dataclass(
    cls_name="OutputEnergyVolumeCurve",
    output_keys=energy_volume_curve_output_keys,
)


OutputElastic = make_output_dataclass(
    cls_name="OutputElastic",
    output_keys=elastic_matrix_output_keys,
)


OutputPhonons = make_output_dataclass(
    cls_name="OutputPhonons",
    output_keys=phonon_output_keys,
)
