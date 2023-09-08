import numpy as np
import scipy.constants
import scipy.optimize

from atomistics.workflows.evcurve import interpolate_energy


def _debye_kernel(xi):
    return xi**3 / (np.exp(xi) - 1)


def debye_integral(x):
    return scipy.integrate.quad(_debye_kernel, 0, x)[0]


def debye_function(x):
    if hasattr(x, "__len__"):
        return np.array([3 / xx**3 * debye_integral(xx) for xx in x])
    return 3 / x**3 * debye_integral(x)


class DebyeModel(object):
    """
    Calculate Thermodynamic Properties based on the Murnaghan output
    """

    def __init__(self, fit_dict, masses, num_steps=50):
        self._fit_dict = fit_dict
        self._masses = masses

        # self._atoms_per_cell = len(murnaghan.structure)
        self._v_min = None
        self._v_max = None
        self._num_steps = None

        self._volume = None
        self._init_volume()

        self.num_steps = num_steps
        self._fit_volume = None
        self._debye_T = None

    def _init_volume(self):
        vol = self._fit_dict["volume"]
        self._v_min, self._v_max = np.min(vol), np.max(vol)

    def _set_volume(self):
        if self._v_min and self._v_max and self._num_steps:
            self._volume = np.linspace(self._v_min, self._v_max, self._num_steps)
            self._reset()

    @property
    def num_steps(self):
        return self._num_steps

    @num_steps.setter
    def num_steps(self, val):
        self._num_steps = val
        self._set_volume()

    @property
    def volume(self):
        if self._volume is None:
            self._init_volume()
            self._set_volume()
        return self._volume

    @volume.setter
    def volume(self, volume_lst):
        self._volume = volume_lst
        self._v_min = np.min(volume_lst)
        self._v_max = np.max(volume_lst)
        self._reset()

    def _reset(self):
        self._debye_T = None

    def interpolate(self, volumes=None):
        if volumes is None:
            volumes = self.volume
        return interpolate_energy(fit_dict=self._fit_dict, volumes=volumes)

    @property
    def debye_temperature(self):
        if self._debye_T is not None:
            return self._debye_T

        GPaTokBar = 10
        Ang3_to_Bohr3 = (
            scipy.constants.angstrom**3
            / scipy.constants.physical_constants["Bohr radius"][0] ** 3
        )
        convert = 67.48  # conversion factor, Moruzzi Eq. (4)
        empirical = 0.617  # empirical factor, Moruzzi Eq. (6)
        gamma_low, gamma_high = 1, 2 / 3  # low/high T gamma

        V0 = self._fit_dict["volume_eq"]
        B0 = self._fit_dict["bulkmodul_eq"]
        Bp = self._fit_dict["b_prime_eq"]

        vol = self.volume

        mass = set(self._masses)
        if len(mass) > 1:
            raise NotImplementedError(
                "Debye temperature only for single species systems!"
            )
        mass = list(mass)[0]

        r0 = (3 * V0 * Ang3_to_Bohr3 / (4 * np.pi)) ** (1.0 / 3.0)
        debye_zero = empirical * convert * np.sqrt(r0 * B0 * GPaTokBar / mass)
        # print('r0, B0, Bp, mass, V0', r0, B0, Bp, mass, V0)
        # print('gamma_low, gamma_high: ', gamma_low, gamma_high)
        # print('debye_zero, V0: ', debye_zero, V0)
        if vol is None:
            print("WARNING: vol: ", vol)

        debye_low = debye_zero * (V0 / vol) ** (-gamma_low + 0.5 * (1 + Bp))
        debye_high = debye_zero * (V0 / vol) ** (-gamma_high + 0.5 * (1 + Bp))

        self._debye_T = (debye_low, debye_high)
        return self._debye_T

    def energy_vib(self, T, debye_T=None, low_T_limit=True):
        kB = 0.086173422 / 1000  # eV/K
        if debye_T is None:
            if low_T_limit:
                debye_T = self.debye_temperature[0]  # low
            else:
                debye_T = self.debye_temperature[1]  # high
        if hasattr(debye_T, "__len__"):
            val = [
                9.0 / 8.0 * kB * d_T
                + T * kB * (3 * np.log(1 - np.exp(-d_T / T)) - debye_function(d_T / T))
                for d_T in debye_T
            ]
            val = np.array(val)
        else:
            val = 9.0 / 8.0 * kB * debye_T + T * kB * (
                3 * np.log(1 - np.exp(-debye_T / T)) - debye_function(debye_T / T)
            )
        atoms_per_cell = len(self._masses)
        return atoms_per_cell * val


def get_debye_model(fit_dict, masses, num_steps=50):
    return DebyeModel(fit_dict=fit_dict, masses=masses, num_steps=num_steps)
