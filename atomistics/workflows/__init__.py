from atomistics.shared.import_warning import raise_warning
from atomistics.workflows.elastic.workflow import ElasticMatrixWorkflow
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

__all__ = [
    "ElasticMatrixWorkflow",
    "EnergyVolumeCurveWorkflow",
    "LangevinWorkflow",
    "calc_molecular_dynamics_thermal_expansion",
    "optimize_cell",
    "optimize_positions",
    "optimize_positions_and_volume",
    "optimize_volume",
]
phonopy_workflows = [
    "PhonopyWorkflow",
    "QuasiHarmonicWorkflow",
]


try:  # in case phonopy is not installed
    from atomistics.workflows.phonons.workflow import PhonopyWorkflow
    from atomistics.workflows.quasiharmonic import QuasiHarmonicWorkflow
except ImportError as e:
    raise_warning(module_list=phonopy_workflows, import_error=e)
else:
    __all__ += phonopy_workflows
