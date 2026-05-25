# best would be StrEnum from py3.11
import sys
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Union

if sys.version_info < (3, 11):
    # official impl' is not significantly different
    class StrEnum(str, Enum):
        def __str__(self):
            return str(self.value)

else:
    from enum import StrEnum


class TaskEnum(StrEnum):
    calc_energy = "calc_energy"
    calc_forces = "calc_forces"
    calc_stress = "calc_stress"
    optimize_positions = "optimize_positions"
    optimize_positions_and_volume = "optimize_positions_and_volume"
    optimize_volume = "optimize_volume"
    optimize_cell = "optimize_cell"
    calc_molecular_dynamics_thermal_expansion = (
        "calc_molecular_dynamics_thermal_expansion"
    )


class TaskOutputEnum(Enum):
    energy = "calc_energy"
    forces = "calc_forces"
    stress = "calc_stress"
    structure_with_optimized_cell = "optimize_cell"
    structure_with_optimized_positions = "optimize_positions"
    structure_with_optimized_positions_and_volume = "optimize_positions_and_volume"
    structure_with_optimized_volume = "optimize_volume"
    volume_over_temperature = "calc_molecular_dynamics_thermal_expansion"


TaskName = Union[str, TaskEnum]


if TYPE_CHECKING:
    from ase import Atoms

    TaskSpec = tuple[Atoms, list[TaskName]]
    TaskDict = dict[str, TaskSpec]

    TaskResults = dict[TaskName, Any]
    ResultsDict = dict[str, TaskResults]

    SimpleEvaluator = Callable[..., TaskResults]


def get_quantities_from_tasks(tasks: Iterable[str]) -> list[str]:
    """
    Get a list of quantities based on the given tasks.

    Args:
        tasks (Iterable[str]): The requested tasks.

    Returns:
        list[str]: A list of quantities.

    """
    quantities = []
    if "calc_energy" in tasks:
        quantities.append("energy")
    if "calc_forces" in tasks:
        quantities.append("forces")
    if "calc_stress" in tasks:
        quantities.append("stress")
    return quantities
