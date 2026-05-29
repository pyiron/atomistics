from abc import ABC, abstractmethod
from typing import Any


class Workflow(ABC):
    """Abstract base class for atomistics simulation workflows.

    Subclasses implement a two-phase protocol: ``generate_structures`` produces
    a task dictionary of structures to evaluate, and ``analyse_structures``
    consumes the calculator output to return the final workflow result.
    """

    @abstractmethod
    def generate_structures(self) -> dict[str, Any]:
        """
        Generate structures for the workflow.

        Returns:
            dict: A dictionary containing the generated structures.
        """
        raise NotImplementedError

    @abstractmethod
    def analyse_structures(self, output_dict: dict[str, Any]) -> Any:
        """
        Analyse the generated structures.

        Args:
            output_dict (dict): A dictionary containing the generated structures.
        """
        raise NotImplementedError
