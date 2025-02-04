import numpy as np
from ase.atoms import Atoms

from atomistics.shared.output import OutputElastic
from atomistics.workflows.elastic.helper import (
    analyse_structures_helper,
    generate_structures_helper,
)
from atomistics.workflows.interface import Workflow


class ElasticMatrixWorkflow(Workflow):
    def __init__(
        self,
        structure: Atoms,
        num_of_point: int = 5,
        eps_range: float = 0.005,
        sqrt_eta: bool = True,
        fit_order: int = 2,
    ):
        """
        Initialize the ElasticMatrixWorkflow object.

        Args:
            structure (Atoms): The atomic structure.
            num_of_point (int, optional): The number of strain points. Defaults to 5.
            eps_range (float, optional): The range of strain. Defaults to 0.005.
            sqrt_eta (bool, optional): Whether to take the square root of eta. Defaults to True.
            fit_order (int, optional): The order of polynomial fit. Defaults to 2.
        """
        self.structure = structure.copy()
        self.num_of_point = num_of_point
        self.eps_range = eps_range
        self.sqrt_eta = sqrt_eta
        self.fit_order = fit_order
        self._data = {}
        self._structure_dict = {}
        self.Lag_strain_list = []
        self.epss = np.array([])
        self.zero_strain_job_name = "s_e_0"

    def generate_structures(self) -> dict:
        """
        Generate the structures for elastic matrix calculation.

        Returns:
            dict: The generated structures.
        """
        self._data, self._structure_dict = generate_structures_helper(
            structure=self.structure,
            eps_range=self.eps_range,
            num_of_point=self.num_of_point,
            zero_strain_job_name=self.zero_strain_job_name,
            sqrt_eta=self.sqrt_eta,
        )
        return {"calc_energy": self._structure_dict}

    def analyse_structures(
        self, output_dict: dict, output_keys: tuple = OutputElastic.keys()
    ) -> dict:
        """
        Analyze the structures and calculate the elastic matrix.

        Args:
            output_dict (dict): The output dictionary.
            output_keys (tuple, optional): The keys to extract from the output dictionary.
                Defaults to OutputElastic.keys().

        Returns:
            dict: The calculated elastic matrix.
        """
        self._data, elastic_dict = analyse_structures_helper(
            output_dict=output_dict,
            sym_dict=self._data,
            fit_order=self.fit_order,
            zero_strain_job_name=self.zero_strain_job_name,
            output_keys=output_keys,
        )
        return elastic_dict
