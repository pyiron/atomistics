from abc import ABC, abstractmethod


class Output:
    def get_output(self, output_keys):
        return {q: getattr(self, q)() for q in output_keys}

    @classmethod
    def keys(cls):
        return tuple(
            [
                k
                for k in cls.__dict__.keys()
                if k[0] != "_" and k not in ["get_output", "keys"]
            ]
        )


class OutputStatic(ABC, Output):
    @abstractmethod
    def forces(self):  # np.ndarray (n, 3) [eV / Ang^2]
        pass

    @abstractmethod
    def energy(self):  # float [eV]
        pass

    @abstractmethod
    def stress(self):  # np.ndarray (3, 3) [GPa]
        pass

    @abstractmethod
    def volume(self):  # float [Ang^3]
        pass


class OutputMolecularDynamics(ABC, Output):
    @abstractmethod
    def positions(self):  # np.ndarray (t, n, 3) [Ang]
        pass

    @abstractmethod
    def cell(self):  # np.ndarray (t, 3, 3) [Ang]
        pass

    @abstractmethod
    def forces(self):  # np.ndarray (n, 3) [eV / Ang^2]
        pass

    @abstractmethod
    def temperature(self):  # np.ndarray (t) [K]
        pass

    @abstractmethod
    def energy_pot(self):  # np.ndarray (t) [eV]
        pass

    @abstractmethod
    def energy_tot(self):  # np.ndarray (t) [eV]
        pass

    @abstractmethod
    def pressure(self):  # np.ndarray (t, 3, 3) [GPa]
        pass

    @abstractmethod
    def velocities(self):  # np.ndarray (t, n, 3) [eV / Ang]
        pass

    @abstractmethod
    def volume(self):  # np.ndarray (t) [Ang^3]
        pass


class OutputThermalExpansion(ABC, Output):
    @abstractmethod
    def temperatures(self):  # np.ndarray (T) [K]
        pass

    @abstractmethod
    def volumes(self):  # np.ndarray (T) [Ang^3]
        pass


class OutputThermodynamic(ABC, Output):
    @abstractmethod
    def temperatures(self):  # np.ndarray (T) [K]
        pass

    @abstractmethod
    def volumes(self):  # np.ndarray (T) [Ang^3]
        pass

    @abstractmethod
    def free_energy(self):  # np.ndarray (T) [eV]
        pass

    @abstractmethod
    def entropy(self):  # np.ndarray (T) [eV]
        pass

    @abstractmethod
    def heat_capacity(self):  # np.ndarray (T) [eV]
        pass


class OutputEnergyVolumeCurve(ABC, Output):
    @abstractmethod
    def energy_eq(self):  # float [eV]
        pass

    @abstractmethod
    def volume_eq(self):  # float [Ang^3]
        pass

    @abstractmethod
    def bulkmodul_eq(self):  # float [GPa]
        pass

    @abstractmethod
    def b_prime_eq(self):  # float
        pass

    @abstractmethod
    def fit_dict(self):  # dict
        pass

    @abstractmethod
    def energy(self):  # np.ndarray (V) [eV]
        pass

    @abstractmethod
    def volume(self):  # np.ndarray (V) [Ang^3]
        pass


class OutputElastic(ABC, Output):
    @abstractmethod
    def elastic_matrix(self):  # np.ndarray (6,6) [GPa]
        pass

    @abstractmethod
    def elastic_matrix_inverse(self):  # np.ndarray (6,6) [GPa]
        pass

    @abstractmethod
    def bulkmodul_voigt(self):  # float [GPa]
        pass

    @abstractmethod
    def bulkmodul_reuss(self):  # float [GPa]
        pass

    @abstractmethod
    def bulkmodul_hill(self):  # float [GPa]
        pass

    @abstractmethod
    def shearmodul_voigt(self):  # float [GPa]
        pass

    @abstractmethod
    def shearmodul_reuss(self):  # float [GPa]
        pass

    @abstractmethod
    def shearmodul_hill(self):  # float [GPa]
        pass

    @abstractmethod
    def youngsmodul_voigt(self):  # float [GPa]
        pass

    @abstractmethod
    def youngsmodul_reuss(self):  # float [GPa]
        pass

    @abstractmethod
    def youngsmodul_hill(self):  # float [GPa]
        pass

    @abstractmethod
    def poissonsratio_voigt(self):  # float
        pass

    @abstractmethod
    def poissonsratio_reuss(self):  # float
        pass

    @abstractmethod
    def poissonsratio_hill(self):  # float
        pass

    @abstractmethod
    def AVR(self):  # float
        pass

    @abstractmethod
    def elastic_matrix_eigval(self):  # np.ndarray (6,6) [GPa]
        pass


class OutputPhonons(ABC, Output):
    @abstractmethod
    def mesh_dict(self):  # dict
        pass

    @abstractmethod
    def band_structure_dict(self):  # dict
        pass

    @abstractmethod
    def total_dos_dict(self):  # dict
        pass

    @abstractmethod
    def dynamical_matrix(self):  # dict
        pass

    @abstractmethod
    def force_constants(self):  # dict
        pass
