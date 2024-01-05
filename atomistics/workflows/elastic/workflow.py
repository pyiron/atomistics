import numpy as np

from atomistics.shared.output import OutputElastic
from atomistics.workflows.interface import Workflow
from atomistics.workflows.elastic.elastic_moduli import ElasticProperties
from atomistics.workflows.elastic.helper import (
    generate_structures_helper,
    analyse_structures_helper,
)


ElasticMatrixOutputElastic = OutputElastic(
    elastic_matrix=ElasticProperties.elastic_matrix,
    elastic_matrix_inverse=ElasticProperties.elastic_matrix_inverse,
    bulkmodul_voigt=ElasticProperties.bulkmodul_voigt,
    bulkmodul_reuss=ElasticProperties.bulkmodul_reuss,
    bulkmodul_hill=ElasticProperties.bulkmodul_hill,
    shearmodul_voigt=ElasticProperties.shearmodul_voigt,
    shearmodul_reuss=ElasticProperties.shearmodul_reuss,
    shearmodul_hill=ElasticProperties.shearmodul_hill,
    youngsmodul_voigt=ElasticProperties.youngsmodul_voigt,
    youngsmodul_reuss=ElasticProperties.youngsmodul_reuss,
    youngsmodul_hill=ElasticProperties.youngsmodul_hill,
    poissonsratio_voigt=ElasticProperties.poissonsratio_voigt,
    poissonsratio_reuss=ElasticProperties.poissonsratio_reuss,
    poissonsratio_hill=ElasticProperties.poissonsratio_hill,
    AVR=ElasticProperties.AVR,
    elastic_matrix_eigval=ElasticProperties.elastic_matrix_eigval,
)


class ElasticMatrixWorkflow(Workflow):
    def __init__(
        self, structure, num_of_point=5, eps_range=0.005, sqrt_eta=True, fit_order=2
    ):
        self.structure = structure.copy()
        self.num_of_point = num_of_point
        self.eps_range = eps_range
        self.sqrt_eta = sqrt_eta
        self.fit_order = fit_order
        self._data = dict()
        self._structure_dict = dict()
        self.Lag_strain_list = []
        self.epss = np.array([])
        self.zero_strain_job_name = "s_e_0"

    def generate_structures(self):
        """

        Returns:

        """
        self._data, self._structure_dict = generate_structures_helper(
            structure=self.structure,
            eps_range=self.eps_range,
            num_of_point=self.num_of_point,
            zero_strain_job_name=self.zero_strain_job_name,
            sqrt_eta=self.sqrt_eta,
        )
        return {"calc_energy": self._structure_dict}

    def analyse_structures(self, output_dict, output=OutputElastic.fields()):
        """

        Args:
            output_dict (dict):
            output (tuple):

        Returns:

        """
        elastic_matrix, A2, strain_energy, ene0 = analyse_structures_helper(
            output_dict=output_dict,
            Lag_strain_list=self._data["Lag_strain_list"],
            epss=self._data["epss"],
            v0=self._data["v0"],
            LC=self._data["LC"],
            fit_order=self.fit_order,
            zero_strain_job_name=self.zero_strain_job_name,
        )
        self._data["strain_energy"] = strain_energy
        self._data["e0"] = ene0
        self._data["A2"] = A2
        return ElasticMatrixOutputElastic.get(
            ElasticProperties(elastic_matrix=elastic_matrix), *output
        )
