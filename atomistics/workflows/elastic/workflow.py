from collections import OrderedDict
import numpy as np
import scipy.constants

from atomistics.shared.output import OutputElastic
from atomistics.workflows.interface import Workflow
from atomistics.workflows.elastic.symmetry import (
    find_symmetry_group_number,
    get_C_from_A2,
    get_LAG_Strain_List,
    get_symmetry_family_from_SGN,
    Ls_Dic,
)
from atomistics.workflows.elastic.elastic_moduli import (
    get_bulkmodul_voigt,
    get_shearmodul_voigt,
    get_youngsmodul_voigt,
    get_poissonsratio_voigt,
    get_bulkmodul_reuss,
    get_shearmodul_reuss,
    get_youngsmodul_reuss,
    get_poissonsratio_reuss,
    get_bulkmodul_hill,
    get_shearmodul_hill,
    get_youngsmodul_hill,
    get_poissonsratio_hill,
    get_AVR,
    get_elastic_matrix_inverse,
    get_elastic_matrix_eigval,
)


class ElasticProperties:
    def __init__(self, elastic_matrix):
        self._elastic_matrix = elastic_matrix
        self._elastic_matrix_inverse = None
        self._bulkmodul_voigt = None
        self._shearmodul_voigt = None
        self._bulkmodul_reuss = None
        self._shearmodul_reuss = None
        self._bulkmodul_hill = None
        self._shearmodul_hill = None

    def get_elastic_matrix(self):
        return self._elastic_matrix

    def get_elastic_matrix_inverse(self):
        if self._elastic_matrix_inverse is None:
            self._elastic_matrix_inverse = get_elastic_matrix_inverse(
                elastic_matrix=self._elastic_matrix
            )
        return self._elastic_matrix_inverse

    def get_bulkmodul_voigt(self):
        if self._bulkmodul_voigt is None:
            self._bulkmodul_voigt = get_bulkmodul_voigt(
                elastic_matrix=self._elastic_matrix
            )
        return self._bulkmodul_voigt

    def get_shearmodul_voigt(self):
        if self._shearmodul_voigt is None:
            self._shearmodul_voigt = get_shearmodul_voigt(
                elastic_matrix=self._elastic_matrix
            )
        return self._shearmodul_voigt

    def get_bulkmodul_reuss(self):
        if self._bulkmodul_reuss is None:
            self._bulkmodul_reuss = get_bulkmodul_reuss(
                elastic_matrix_inverse=self.get_elastic_matrix_inverse()
            )
        return self._bulkmodul_reuss

    def get_shearmodul_reuss(self):
        if self._shearmodul_reuss is None:
            self._shearmodul_reuss = get_shearmodul_reuss(
                elastic_matrix_inverse=self.get_elastic_matrix_inverse()
            )
        return self._shearmodul_reuss

    def get_bulkmodul_hill(self):
        if self._bulkmodul_hill is None:
            self._bulkmodul_hill = get_bulkmodul_hill(
                bulkmodul_voigt=self.get_bulkmodul_voigt(),
                bulkmodul_reuss=self.get_bulkmodul_reuss(),
            )
        return self._bulkmodul_hill

    def get_shearmodul_hill(self):
        if self._shearmodul_hill is None:
            self._shearmodul_hill = get_shearmodul_hill(
                shearmodul_voigt=self.get_shearmodul_voigt(),
                shearmodul_reuss=self.get_shearmodul_reuss(),
            )
        return self._shearmodul_hill

    def get_youngsmodul_voigt(self):
        return get_youngsmodul_voigt(
            bulkmodul_voigt=self.get_bulkmodul_voigt(),
            shearmodul_voigt=self.get_shearmodul_voigt(),
        )

    def get_poissonsratio_voigt(self):
        return get_poissonsratio_voigt(
            bulkmodul_voigt=self.get_bulkmodul_voigt(),
            shearmodul_voigt=self.get_shearmodul_voigt(),
        )

    def get_youngsmodul_reuss(self):
        return get_youngsmodul_reuss(
            bulkmodul_reuss=self.get_bulkmodul_reuss(),
            shearmodul_reuss=self.get_shearmodul_reuss(),
        )

    def get_poissonsratio_reuss(self):
        return get_poissonsratio_reuss(
            bulkmodul_reuss=self.get_bulkmodul_reuss(),
            shearmodul_reuss=self.get_shearmodul_reuss(),
        )

    def get_youngsmodul_hill(self):
        return get_youngsmodul_hill(
            bulkmodul_hill=self.get_bulkmodul_hill(),
            shearmodul_hill=self.get_shearmodul_hill(),
        )

    def get_poissonratio_hill(self):
        return get_poissonsratio_hill(
            bulkmodul_hill=self.get_bulkmodul_hill(),
            shearmodul_hill=self.get_shearmodul_hill(),
        )

    def get_AVR(self):
        return get_AVR(
            shearmodul_voigt=self.get_shearmodul_voigt(),
            shearmodul_reuss=self.get_shearmodul_reuss(),
        )

    def get_elastic_matrix_eigval(self):
        return get_elastic_matrix_eigval(elastic_matrix=self._elastic_matrix)


ElasticMatrixOutputElastic = OutputElastic(
    elastic_matrix=ElasticProperties.get_elastic_matrix,
    elastic_matrix_inverse=ElasticProperties.get_elastic_matrix_inverse,
    bulkmodul_voigt=ElasticProperties.get_bulkmodul_voigt,
    bulkmodul_reuss=ElasticProperties.get_bulkmodul_reuss,
    bulkmodul_hill=ElasticProperties.get_bulkmodul_hill,
    shearmodul_voigt=ElasticProperties.get_shearmodul_voigt,
    shearmodul_reuss=ElasticProperties.get_shearmodul_reuss,
    shearmodul_hill=ElasticProperties.get_shearmodul_hill,
    youngsmodul_voigt=ElasticProperties.get_youngsmodul_voigt,
    youngsmodul_reuss=ElasticProperties.get_youngsmodul_reuss,
    youngsmodul_hill=ElasticProperties.get_youngsmodul_hill,
    poissonsratio_voigt=ElasticProperties.get_poissonsratio_voigt,
    poissonsratio_reuss=ElasticProperties.get_poissonsratio_reuss,
    poissonsratio_hill=ElasticProperties.get_poissonratio_hill,
    AVR=ElasticProperties.get_AVR,
    elastic_matrix_eigval=ElasticProperties.get_elastic_matrix_eigval,
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
        self._data = OrderedDict()
        self._structure_dict = OrderedDict()
        self.SGN = None
        self.v0 = None
        self.LC = None
        self.Lag_strain_list = []
        self.epss = np.array([])
        self.zero_strain_job_name = "s_e_0"

    def symmetry_analysis(self):
        """

        Returns:

        """
        self.SGN = find_symmetry_group_number(self.structure)
        self._data["SGN"] = self.SGN
        self.v0 = self.structure.get_volume()
        self._data["v0"] = self.v0
        self.LC = get_symmetry_family_from_SGN(self.SGN)
        self._data["LC"] = self.LC
        self.Lag_strain_list = get_LAG_Strain_List(self.LC)
        self._data["Lag_strain_list"] = self.Lag_strain_list
        self.epss = np.linspace(-self.eps_range, self.eps_range, self.num_of_point)
        self._data["epss"] = self.epss

    def generate_structures(self):
        """

        Returns:

        """
        self.symmetry_analysis()
        basis_ref = self.structure
        Lag_strain_list = self.Lag_strain_list
        epss = self.epss

        if 0.0 in epss:
            self._structure_dict[self.zero_strain_job_name] = basis_ref.copy()

        for lag_strain in Lag_strain_list:
            Ls_list = Ls_Dic[lag_strain]
            for eps in epss:
                if eps == 0.0:
                    continue

                Ls = np.zeros(6)
                for ii in range(6):
                    Ls[ii] = Ls_list[ii]
                Lv = eps * Ls

                eta_matrix = np.zeros((3, 3))

                eta_matrix[0, 0] = Lv[0]
                eta_matrix[0, 1] = Lv[5] / 2.0
                eta_matrix[0, 2] = Lv[4] / 2.0

                eta_matrix[1, 0] = Lv[5] / 2.0
                eta_matrix[1, 1] = Lv[1]
                eta_matrix[1, 2] = Lv[3] / 2.0

                eta_matrix[2, 0] = Lv[4] / 2.0
                eta_matrix[2, 1] = Lv[3] / 2.0
                eta_matrix[2, 2] = Lv[2]

                norm = 1.0
                eps_matrix = eta_matrix
                if np.linalg.norm(eta_matrix) > 0.7:
                    raise Exception("Too large deformation %g" % eps)

                if self.sqrt_eta:
                    while norm > 1.0e-10:
                        x = eta_matrix - np.dot(eps_matrix, eps_matrix) / 2.0
                        norm = np.linalg.norm(x - eps_matrix)
                        eps_matrix = x

                # --- Calculating the M_new matrix ---------------------------------------------------------
                i_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                def_matrix = i_matrix + eps_matrix
                scell = np.dot(basis_ref.get_cell(), def_matrix)
                nstruct = basis_ref.copy()
                nstruct.set_cell(scell, scale_atoms=True)

                jobname = self.subjob_name(lag_strain, eps)

                self._structure_dict[jobname] = nstruct

        return {"calc_energy": self._structure_dict}

    def analyse_structures(self, output_dict, output=OutputElastic.fields()):
        """

        Args:
            output_dict (dict):
            output (tuple):

        Returns:

        """
        self.symmetry_analysis()
        epss = self.epss
        Lag_strain_list = self.Lag_strain_list

        if "energy" in output_dict.keys():
            output_dict = output_dict["energy"]

        ene0 = None
        if 0.0 in epss:
            ene0 = output_dict[self.zero_strain_job_name]
        self._data["e0"] = ene0
        strain_energy = []
        for lag_strain in Lag_strain_list:
            strain_energy.append([])
            for eps in epss:
                if not eps == 0.0:
                    jobname = self.subjob_name(lag_strain, eps)
                    ene = output_dict[jobname]
                else:
                    ene = ene0
                strain_energy[-1].append((eps, ene))
        self._data["strain_energy"] = strain_energy
        self.fit_elastic_matrix()
        return ElasticMatrixOutputElastic.get(
            ElasticProperties(elastic_matrix=self._data["C"]), *output
        )

    def fit_elastic_matrix(self):
        """

        Returns:

        """
        strain_ene = self._data["strain_energy"]

        v0 = self._data["v0"]
        LC = self._data["LC"]
        A2 = []
        fit_order = int(self.fit_order)
        for s_e in strain_ene:
            ss = np.transpose(s_e)
            coeffs = np.polyfit(ss[0], ss[1] / v0, fit_order)
            A2.append(coeffs[fit_order - 2])

        A2 = np.array(A2)
        C = get_C_from_A2(A2, LC)

        for i in range(5):
            for j in range(i + 1, 6):
                C[j, i] = C[i, j]

        CONV = (
            1e21
            / scipy.constants.physical_constants["joule-electron volt relationship"][0]
        )  # From eV/Ang^3 to GPa

        C *= CONV
        self._data["C"] = C
        self._data["A2"] = A2

    @staticmethod
    def subjob_name(i, eps):
        """

        Args:
            i:
            eps:

        Returns:

        """
        return ("s_%s_e_%.5f" % (i, eps)).replace(".", "_").replace("-", "m")
