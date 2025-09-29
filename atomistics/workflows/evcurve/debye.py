import numpy as np
import scipy.constants
import scipy.optimize

from atomistics.shared.output import OutputThermodynamic
from atomistics.workflows.evcurve.fit import interpolate_energy
from atomistics.workflows.evcurve.thermo import get_thermo_bulk_model


class DebyeThermalProperties:
    def __init__(
        self,
        fit_dict: dict,
        masses: list[float],
        t_min: float = 1.0,
        t_max: float = 1500.0,
        t_step: float = 50.0,
        temperatures: np.ndarray = None,
        constant_volume: bool = False,
        num_steps: int = 50,
    ):
        """
        Initialize the DebyeThermalProperties class.

        Parameters:
        - fit_dict (dict): The fit dictionary containing the volume and bulk modulus information.
        - masses (list[float]): The masses of the atoms in the system.
        - t_min (float): The minimum temperature in Kelvin. Default is 1.0.
        - t_max (float): The maximum temperature in Kelvin. Default is 1500.0.
        - t_step (float): The temperature step size in Kelvin. Default is 50.0.
        - temperatures (np.ndarray): The array of temperatures. If None, it will be generated based on t_min, t_max, and t_step. Default is None.
        - constant_volume (bool): Whether to calculate properties at constant volume. Default is False.
        - num_steps (int): The number of steps for volume interpolation. Default is 50.
        """
        if temperatures is None:
            temperatures = np.arange(t_min, t_max + t_step, t_step)
        self._temperatures = temperatures
        self._debye_model = get_debye_model(
            fit_dict=fit_dict, masses=masses, num_steps=num_steps
        )
        self._pes = get_thermo_bulk_model(
            temperatures=temperatures,
            debye_model=self._debye_model,
        )
        self._constant_volume = constant_volume

    def free_energy(self) -> np.ndarray:
        """
        Calculate the free energy.

        Returns:
        - np.ndarray: The array of free energy values.
        """
        return (
            self._pes.get_free_energy_p()
            - self._debye_model.interpolate(volumes=self._pes.get_minimum_energy_path())
        ) / self._pes.num_atoms

    def temperatures(self) -> np.ndarray:
        """
        Get the array of temperatures.

        Returns:
        - np.ndarray: The array of temperatures.
        """
        return self._temperatures

    def entropy(self) -> np.ndarray:
        """
        Calculate the entropy.

        Returns:
        - np.ndarray: The array of entropy values.
        """
        if not self._constant_volume:
            return (
                self._pes.eV_to_J_per_mol
                / self._pes.num_atoms
                * self._pes.get_entropy_p()
            )
        else:
            return (
                self._pes.eV_to_J_per_mol
                / self._pes.num_atoms
                * self._pes.get_entropy_v()
            )

    def heat_capacity(self) -> np.ndarray:
        """
        Calculate the heat capacity.

        Returns:
        - np.ndarray: The array of heat capacity values.
        """
        if not self._constant_volume:
            heat_capacity = (
                self._pes.eV_to_J_per_mol
                * self._pes.temperatures[:-2]
                * np.gradient(self._pes.get_entropy_p(), self._pes._d_temp)[:-2]
            )
        else:
            heat_capacity = (
                self._pes.eV_to_J_per_mol
                * self._pes.temperatures[:-2]
                * np.gradient(self._pes.get_entropy_v(), self._pes._d_temp)[:-2]
            )
        return np.array(heat_capacity.tolist() + [np.nan, np.nan])

    def volumes(self) -> np.ndarray:
        """
        Get the array of volumes.

        Returns:
        - np.ndarray: The array of volumes.
        """
        if not self._constant_volume:
            return self._pes.get_minimum_energy_path()
        else:
            return np.array([self._pes.volumes[0]] * len(self._temperatures))


def _debye_kernel(xi: np.ndarray) -> np.ndarray:
    """
    Calculate the Debye kernel.

    Parameters:
    - xi (np.ndarray): The array of values.

    Returns:
    - np.ndarray: The array of Debye kernel values.
    """
    return xi**3 / (np.exp(xi) - 1)


def debye_integral(x: np.ndarray) -> np.ndarray:
    """
    Calculate the Debye integral for a given array of values.

    Parameters:
        x (np.ndarray): Array of values for which the Debye integral is calculated.

    Returns:
        np.ndarray: Array of Debye integral values corresponding to the input values.
    """
    return scipy.integrate.quad(_debye_kernel, 0, x)[0]


def debye_function(x: np.ndarray) -> np.ndarray:
    """
    Calculate the Debye function for a given array of values.

    Parameters:
        x (np.ndarray): Array of values for which the Debye function is calculated.

    Returns:
        np.ndarray: Array of Debye function values corresponding to the input values.
    """
    if hasattr(x, "__len__"):
        return np.array([3 / xx**3 * debye_integral(xx) for xx in x])
    return 3 / x**3 * debye_integral(x)


class DebyeModel:
    """
    Calculate Thermodynamic Properties based on the Murnaghan output
    """

    def __init__(self, fit_dict: dict, masses: list[float], num_steps: int = 50):
        """
        Initialize the DebyeModel class.

        Parameters:
        - fit_dict (dict): The fit dictionary containing the volume and bulk modulus information.
        - masses (list[float]): The masses of the atoms in the system.
        - num_steps (int): The number of steps for volume interpolation. Default is 50.
        """
        self._fit_dict = fit_dict
        self._masses = masses

        self._v_min = None
        self._v_max = None
        self._num_steps = None

        self._volume = None
        self._init_volume()

        self.num_steps = num_steps
        self._fit_volume = None
        self._debye_T = None

    def _init_volume(self):
        """
        Initialize the minimum and maximum volume values.
        """
        vol = self._fit_dict["volume"]
        self._v_min, self._v_max = np.min(vol), np.max(vol)

    def _set_volume(self):
        """
        Set the volume array based on the minimum and maximum volume values.
        """
        if self._v_min and self._v_max and self._num_steps:
            self._volume = np.linspace(self._v_min, self._v_max, self._num_steps)
            self._reset()

    @property
    def num_steps(self) -> int:
        """
        Get the number of steps for volume interpolation.

        Returns:
        - int: The number of steps.
        """
        return self._num_steps

    @num_steps.setter
    def num_steps(self, val: int):
        """
        Set the number of steps for volume interpolation.

        Parameters:
        - val (int): The number of steps.
        """
        self._num_steps = val
        self._set_volume()

    @property
    def volume(self) -> np.ndarray:
        """
        Get the array of volumes.

        Returns:
        - np.ndarray: The array of volumes.
        """
        if self._volume is None:
            self._init_volume()
            self._set_volume()
        return self._volume

    @volume.setter
    def volume(self, volume_lst: np.ndarray):
        """
        Set the array of volumes.

        Parameters:
        - volume_lst (np.ndarray): The array of volumes.
        """
        self._volume = volume_lst
        self._v_min = np.min(volume_lst)
        self._v_max = np.max(volume_lst)
        self._reset()

    def _reset(self):
        """
        Reset the Debye temperature.
        """
        self._debye_T = None

    def interpolate(self, volumes: np.ndarray = None) -> np.ndarray:
        """
        Interpolate the energy based on the fit dictionary and volumes.

        Parameters:
        - volumes (np.ndarray): The array of volumes. If None, use the volume array of the DebyeModel.

        Returns:
        - np.ndarray: The interpolated energy values.
        """
        if volumes is None:
            volumes = self.volume
        return interpolate_energy(fit_dict=self._fit_dict, volumes=volumes)

    @property
    def debye_temperature(self) -> tuple[float]:
        """
        Get the Debye temperature.

        Returns:
        - tuple[float]: The Debye temperature values.
        """
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

        if vol is None:
            print("WARNING: vol: ", vol)

        debye_low = debye_zero * (V0 / vol) ** (-gamma_low + 0.5 * (1 + Bp))
        debye_high = debye_zero * (V0 / vol) ** (-gamma_high + 0.5 * (1 + Bp))

        self._debye_T = (debye_low, debye_high)
        return self._debye_T

    def energy_vib(
        self, T: np.ndarray, debye_T: tuple[float] = None, low_T_limit: bool = True
    ):
        """
        Calculate the vibrational energy.

        Parameters:
        - T (np.ndarray): The array of temperatures.
        - debye_T (tuple[float]): The Debye temperature values. If None, use the Debye temperature of the DebyeModel.
        - low_T_limit (bool): Whether to use the low temperature limit. Default is True.

        Returns:
        - np.ndarray: The array of vibrational energy values.
        """
        kB = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
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


def get_debye_model(
    fit_dict: dict[str, float], masses: list[float], num_steps: int = 50
) -> DebyeModel:
    """
    Create a DebyeModel object with the given parameters.

    Args:
        fit_dict (Dict[str, float]): A dictionary containing the fit parameters.
        masses (List[float]): A list of masses for the atoms in the system.
        num_steps (int, optional): The number of steps to use in the Debye model. Defaults to 50.

    Returns:
        DebyeModel: A DebyeModel object initialized with the given parameters.
    """
    return DebyeModel(fit_dict=fit_dict, masses=masses, num_steps=num_steps)


def get_thermal_properties_for_energy_volume_curve(
    fit_dict: dict,
    masses: list[float],
    t_min: float = 1.0,
    t_max: float = 1500.0,
    t_step: float = 50.0,
    temperatures: np.ndarray = None,
    constant_volume: bool = False,
    num_steps: int = 50,
    output_keys: tuple = OutputThermodynamic.keys(),
) -> dict:
    """
    Calculate the thermal properties based on the Debye model.

    Parameters:
    - fit_dict (dict): The fit dictionary containing the volume and bulk modulus information.
    - masses (list[float]): The masses of the atoms in the system.
    - t_min (float): The minimum temperature. Default is 1.0.
    - t_max (float): The maximum temperature. Default is 1500.0.
    - t_step (float): The temperature step. Default is 50.0.
    - temperatures (np.ndarray): The array of temperatures. If None, it will be generated based on t_min, t_max, and t_step.
    - constant_volume (bool): Whether to calculate the properties at constant volume. Default is False.
    - num_steps (int): The number of steps for volume interpolation. Default is 50.
    - output_keys (tuple): The keys of the output properties to include in the result. Default is OutputThermodynamic.keys().

    Returns:
    - dict: A dictionary containing the calculated thermal properties.
    """
    debye_model = DebyeThermalProperties(
        fit_dict=fit_dict,
        masses=masses,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        temperatures=temperatures,
        constant_volume=constant_volume,
        num_steps=num_steps,
    )
    return OutputThermodynamic(
        **{k: getattr(debye_model, k) for k in OutputThermodynamic.keys()}
    ).get(output_keys=output_keys)
