from enum import Enum
from typing import NewType, Union, TYPE_CHECKING

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

if TYPE_CHECKING:
    TaskName = NewType("TaskName", Union[str, TaskEnum])
    TaskSpec = NewType("TaskSpec", tuple[Atoms, list[TaskName]])
    TaskDict = NewType("TaskDict", dict[str, TaskSpec])

    TaskResults = NewType("TaskResults", dict[TaskName, Any])
    ResultsDict = NewType("ResultsDict", dict[str, TaskResults])
