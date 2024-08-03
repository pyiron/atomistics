from abc import ABC


class Workflow(ABC):
    def generate_structures(self) -> dict:
        """
        Generate structures for the workflow.

        Returns:
            A dictionary containing the generated structures.
        """
        raise NotImplementedError

    def analyse_structures(self, output_dict: dict):
        """
        Analyse the generated structures.

        Args:
            output_dict: A dictionary containing the generated structures.
        """
        raise NotImplementedError
