from abc import ABC, abstractmethod


class Workflow(ABC):
    @abstractmethod
    def generate_structures(self) -> dict:
        """
        Generate structures for the workflow.

        Returns:
            dict: A dictionary containing the generated structures.
        """
        raise NotImplementedError

    @abstractmethod
    def analyse_structures(self, output_dict: dict) -> dict:
        """
        Analyse the generated structures.

        Args:
            output_dict (dict): A dictionary containing the generated structures.
        """
        raise NotImplementedError
