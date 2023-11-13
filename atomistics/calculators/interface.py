from enum import Enum
from typing import NewType, Union, Any, TYPE_CHECKING

# best would be StrEnum from py3.11
import sys

if sys.version_info.minor < 11:
    # official impl' is not significantly different
    class StrEnum(str, Enum):
        def __str__(self):
            return str(self.value)

else:
    from enum import StrEnum


class TaskEnum(StrEnum):
    calc_energy = "calc_energy"
    calc_forces = "calc_forces"
    optimize_positions = "optimize_positions"
    optimize_positions_and_volume = "optimize_positions_and_volume"


class TaskOutputEnum(Enum):
    energy = "calc_energy"
    forces = "calc_forces"
    structure_with_optimized_positions = "optimize_positions"
    structure_with_optimized_positions_and_volume = "optimize_positions_and_volume"


if TYPE_CHECKING:
    from ase import Atoms

    TaskName = Union[str, TaskEnum]
    TaskSpec = tuple[Atoms, list[TaskName]]
    TaskDict = dict[str, TaskSpec]

    TaskResults = dict[TaskName, Any]
    ResultsDict = dict[str, TaskResults]

    SimpleEvaluator = callable[[Atoms, list[TaskName], ...], TaskResults]
