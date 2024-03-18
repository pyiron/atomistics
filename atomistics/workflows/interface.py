from abc import ABC


class Workflow(ABC):
    def generate_structures(self) -> dict:
        raise NotImplementedError

    def analyse_structures(self, output_dict: dict):
        raise NotImplementedError
