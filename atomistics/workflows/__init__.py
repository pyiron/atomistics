from atomistics.shared.import_warning import raise_warning
from atomistics.workflows.elastic.workflow import (
    ElasticMatrixWorkflow,
    analyse_results_for_elastic_matrix,
    get_tasks_for_elastic_matrix,
)
from atomistics.workflows.evcurve.debye import (
    get_thermal_properties_for_energy_volume_curve,
)
from atomistics.workflows.evcurve.helper import (
    analyse_results_for_energy_volume_curve,
    get_tasks_for_energy_volume_curve,
)
from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow
from atomistics.workflows.langevin import LangevinWorkflow
from atomistics.workflows.molecular_dynamics import (
    calc_molecular_dynamics_thermal_expansion,
)
from atomistics.workflows.structure_optimization import (
    optimize_cell,
    optimize_positions,
    optimize_positions_and_volume,
    optimize_volume,
)

__all__: list[str] = [
    "ElasticMatrixWorkflow",
    "EnergyVolumeCurveWorkflow",
    "LangevinWorkflow",
    "analyse_results_for_elastic_matrix",
    "analyse_results_for_energy_volume_curve",
    "calc_molecular_dynamics_thermal_expansion",
    "get_tasks_for_elastic_matrix",
    "get_tasks_for_energy_volume_curve",
    "get_thermal_properties_for_energy_volume_curve",
    "optimize_cell",
    "optimize_positions",
    "optimize_positions_and_volume",
    "optimize_volume",
]
phonopy_workflows: list[str] = [
    "PhonopyWorkflow",
    "QuasiHarmonicWorkflow",
    "get_band_structure",
    "get_dynamical_matrix",
    "get_hesse_matrix",
    "get_thermal_properties_for_harmonic_approximation",
    "get_tasks_for_harmonic_approximation",
    "analyse_results_for_harmonic_approximation",
    "get_tasks_for_quasi_harmonic_approximation",
    "analyse_results_for_quasi_harmonic_approximation",
    "get_thermal_properties_for_quasi_harmonic_approximation",
    "plot_band_structure",
    "plot_dos",
]


try:  # in case phonopy is not installed
    from atomistics.workflows.phonons.helper import (
        analyse_results_for_harmonic_approximation,
        get_band_structure,
        get_dynamical_matrix,
        get_hesse_matrix,
        get_tasks_for_harmonic_approximation,
        get_thermal_properties_for_harmonic_approximation,
        plot_band_structure,
        plot_dos,
    )
    from atomistics.workflows.phonons.workflow import PhonopyWorkflow
    from atomistics.workflows.quasiharmonic import (
        QuasiHarmonicWorkflow,
        analyse_results_for_quasi_harmonic_approximation,
        get_tasks_for_quasi_harmonic_approximation,
        get_thermal_properties_for_quasi_harmonic_approximation,
    )
except ImportError as e:
    raise_warning(module_list=phonopy_workflows, import_error=e)
else:
    __all__ += phonopy_workflows
