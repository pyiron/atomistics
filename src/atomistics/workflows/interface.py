from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class Workflow(ABC):
    @abstractmethod
    def generate_structures(self) -> dict[str, Any]:
        """
        Generate structures for the workflow.

        Returns:
            dict: A dictionary containing the generated structures.
        """
        raise NotImplementedError

    @abstractmethod
    def analyse_structures(self, output_dict: Mapping[str, Any]) -> Any:
        """
        Analyse the generated structures.

        Args:
            output_dict (dict): A dictionary containing the generated structures.
        """
        raise NotImplementedError
