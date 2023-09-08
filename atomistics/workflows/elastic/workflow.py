from collections import OrderedDict
import numpy as np
import scipy.constants

from atomistics.workflows.shared.workflow import Workflow
from atomistics.workflows.elastic.symmetry import (
    find_symmetry_group_number,
    get_C_from_A2,
    get_LAG_Strain_List,
    get_symmetry_family_from_SGN,
    Ls_Dic,
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

    def analyse_structures(self, output_dict):
        """

        Args:
            output_dict (dict):

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
        return self._data

    def calculate_modulus(self):
        """

        Returns:

        """
        C = self._data["C"]

        BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
        GV = (
            (C[0, 0] + C[1, 1] + C[2, 2])
            - (C[0, 1] + C[0, 2] + C[1, 2])
            + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
        ) / 15
        EV = (9 * BV * GV) / (3 * BV + GV)
        nuV = (1.5 * BV - GV) / (3 * BV + GV)
        self._data["BV"] = BV
        self._data["GV"] = GV
        self._data["EV"] = EV
        self._data["nuV"] = nuV

        try:
            S = np.linalg.inv(C)

            BR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
            GR = 15 / (
                4 * (S[0, 0] + S[1, 1] + S[2, 2])
                - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
            )
            ER = (9 * BR * GR) / (3 * BR + GR)
            nuR = (1.5 * BR - GR) / (3 * BR + GR)

            BH = 0.50 * (BV + BR)
            GH = 0.50 * (GV + GR)
            EH = (9.0 * BH * GH) / (3.0 * BH + GH)
            nuH = (1.5 * BH - GH) / (3.0 * BH + GH)

            AVR = 100.0 * (GV - GR) / (GV + GR)
            self._data["S"] = S

            self._data["BR"] = BR
            self._data["GR"] = GR
            self._data["ER"] = ER
            self._data["nuR"] = nuR

            self._data["BH"] = BH
            self._data["GH"] = GH
            self._data["EH"] = EH
            self._data["nuH"] = nuH

            self._data["AVR"] = AVR
        except np.linalg.LinAlgError as e:
            print("LinAlgError:", e)

        eigval = np.linalg.eig(C)
        self._data["C_eigval"] = eigval

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
        self.calculate_modulus()

    @staticmethod
    def subjob_name(i, eps):
        """

        Args:
            i:
            eps:

        Returns:

        """
        return ("s_%s_e_%.5f" % (i, eps)).replace(".", "_").replace("-", "m")
