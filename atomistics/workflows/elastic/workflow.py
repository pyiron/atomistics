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
from atomistics.workflows.elastic.material import (
    get_BV,
    get_GV,
    get_EV,
    get_nuV,
    get_BR,
    get_GR,
    get_ER,
    get_nuR,
    get_BH,
    get_GH,
    get_EH,
    get_nuH,
    get_AVR,
    get_S,
    get_C_eigval,
)


class ElasticProperties:
    def __init__(self, C):
        self._C = C
        self._S = None
        self._BV = None
        self._GV = None
        self._BR = None
        self._GR = None
        self._BH = None
        self._GH = None

    def get_C(self):
        return self._C

    def get_S(self):
        if self._S is None:
            self._S = get_S(C=self._C)
        return self._S

    def get_BV(self):
        if self._BV is None:
            self._BV = get_BV(C=self._C)
        return self._BV

    def get_GV(self):
        if self._GV is None:
            self._GV = get_GV(C=self._C)
        return self._GV

    def get_BR(self):
        if self._BR is None:
            self._BR = get_BR(S=self.get_S())
        return self._BR

    def get_GR(self):
        if self._GR is None:
            self._GR = get_GR(S=self.get_S())
        return self._GR

    def get_BH(self):
        if self._BH is None:
            self._BH = get_BH(BV=self.get_BV(), BR=self.get_BR())
        return self._BH

    def get_GH(self):
        if self._GH is None:
            self._GH = get_GH(GV=self.get_GV(), GR=self.get_GR())
        return self._GH

    def get_EV(self):
        return get_EV(BV=self.get_BV(), GV=self.get_GV())

    def get_nuV(self):
        return get_nuV(BV=self.get_BV(), GV=self.get_GV())

    def get_ER(self):
        return get_ER(BR=self.get_BR(), GR=self.get_GR())

    def get_nuR(self):
        return get_nuR(BR=self.get_BR(), GR=self.get_GR())

    def get_EH(self):
        return get_EH(BH=self.get_BH(), GH=self.get_GH())

    def get_nuH(self):
        return get_nuH(BH=self.get_BH(), GH=self.get_GH())

    def get_AVR(self):
        return get_AVR(GV=self.get_GV(), GR=self.get_GR())

    def get_C_eigval(self):
        return get_C_eigval(C=self._C)


ElasticMatrixOutputElastic = OutputElastic(
    C=ElasticProperties.get_C,
    S=ElasticProperties.get_S,
    BV=ElasticProperties.get_BV,
    BR=ElasticProperties.get_BR,
    BH=ElasticProperties.get_BH,
    GV=ElasticProperties.get_GV,
    GR=ElasticProperties.get_GR,
    GH=ElasticProperties.get_GH,
    EV=ElasticProperties.get_EV,
    ER=ElasticProperties.get_ER,
    EH=ElasticProperties.get_EH,
    nuV=ElasticProperties.get_nuV,
    nuR=ElasticProperties.get_nuR,
    nuH=ElasticProperties.get_nuH,
    AVR=ElasticProperties.get_AVR,
    C_eigval=ElasticProperties.get_C_eigval,
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
            ElasticProperties(C=self._data["C"]), *output
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
