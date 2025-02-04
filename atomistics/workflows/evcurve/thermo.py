from copy import copy
from typing import Optional

import numpy as np
import scipy.constants


class ThermoBulk:
    """
    Class should provide all tools to compute bulk thermodynamic quantities. Central quantity is the Free Energy F(V,T).
    ToDo: Make it a (light weight) pyiron object (introduce a new tool rather than job object).
    """

    eV_to_J_per_mol = scipy.constants.electron_volt * scipy.constants.Avogadro
    kB = 1 / scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

    def __init__(self):
        self._volumes: Optional[np.ndarray] = None
        self._temperatures: Optional[np.ndarray] = None
        self._energies: Optional[np.ndarray] = None
        self._entropy: Optional[np.ndarray] = None
        self._pressure: Optional[np.ndarray] = None
        self._num_atoms: Optional[int] = None

        self._fit_order = 3

    def copy(self) -> "ThermoBulk":
        """
        Returns:
            A copy of the ThermoBulk object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__init__()
        result.__dict__["_volumes"] = copy(self._volumes)
        result.__dict__["_temperatures"] = copy(self._temperatures)
        result.__dict__["_energies"] = copy(self._energies)
        result.__dict__["_fit_order"] = self._fit_order
        return result

    def _reset_energy(self):
        """
        Reset the energy array.
        """
        if self._volumes is not None and self._temperatures is not None:
            self._energies = np.zeros((len(self._temperatures), len(self._volumes)))

    @property
    def num_atoms(self) -> int:
        """
        Get the number of atoms.

        Returns:
            The number of atoms.
        """
        if self._num_atoms is None:
            return 1  # normalize per cell if number of atoms unknown
        return self._num_atoms

    @num_atoms.setter
    def num_atoms(self, num: int):
        """
        Set the number of atoms.

        Args:
            num: The number of atoms.
        """
        self._num_atoms = num

    @property
    def _coeff(self) -> np.ndarray:
        """
        Get the coefficients of the polynomial fit.

        Returns:
            The coefficients of the polynomial fit.
        """
        return np.polyfit(self._volumes, self._energies.T, deg=self._fit_order)

    @property
    def temperatures(self) -> np.ndarray:
        """
        Get the temperatures.

        Returns:
            The temperatures.
        """
        return self._temperatures

    @property
    def _d_temp(self) -> float:
        """
        Get the temperature step size.

        Returns:
            The temperature step size.
        """
        return self.temperatures[1] - self.temperatures[0]

    @property
    def _d_vol(self) -> float:
        """
        Get the volume step size.

        Returns:
            The volume step size.
        """
        return self.volumes[1] - self.volumes[0]

    @temperatures.setter
    def temperatures(self, temp_lst: np.ndarray):
        """
        Set the temperatures.

        Args:
            temp_lst: The temperatures.
        """
        if not hasattr(temp_lst, "__len__"):
            raise ValueError("Requires list as input parameter")
        len_temp = -1
        if self._temperatures is not None:
            len_temp = len(self._temperatures)
        self._temperatures = np.array(temp_lst)
        if len(temp_lst) != len_temp:
            self._reset_energy()

    @property
    def volumes(self) -> np.ndarray:
        """
        Get the volumes.

        Returns:
            The volumes.
        """
        return self._volumes

    @volumes.setter
    def volumes(self, volume_lst: np.ndarray):
        """
        Set the volumes.

        Args:
            volume_lst: The volumes.
        """
        if not hasattr(volume_lst, "__len__"):
            raise ValueError("Requires list as input parameter")
        len_vol = -1
        if self._volumes is not None:
            len_vol = len(self._volumes)
        self._volumes = np.array(volume_lst)
        if len(volume_lst) != len_vol:
            self._reset_energy()

    @property
    def entropy(self) -> np.ndarray:
        """
        Get the entropy.

        Returns:
            The entropy.
        """
        if self._entropy is None:
            self._compute_thermo()
        return self._entropy

    @property
    def pressure(self) -> np.ndarray:
        """
        Get the pressure.

        Returns:
            The pressure.
        """
        if self._pressure is None:
            self._compute_thermo()
        return self._pressure

    @property
    def energies(self) -> np.ndarray:
        """
        Get the energies.

        Returns:
            The energies.
        """
        return self._energies

    @energies.setter
    def energies(self, erg_lst: np.ndarray):
        """
        Set the energies.

        Args:
            erg_lst: The energies.
        """
        if np.ndim(erg_lst) == 2:
            self._energies = erg_lst
        elif np.ndim(erg_lst) == 1:
            if len(erg_lst) == len(self.volumes):
                self._energies = np.tile(erg_lst, (len(self.temperatures), 1))
            else:
                raise ValueError()
        else:
            self._energies = (
                np.ones((len(self.volumes), len(self.temperatures))) * erg_lst
            )

    def set_temperatures(
        self,
        temperature_min: float = 0.0,
        temperature_max: float = 1500.0,
        temperature_steps: float = 50.0,
    ):
        """
        Set the temperatures.

        Args:
            temperature_min: The minimum temperature.
            temperature_max: The maximum temperature.
            temperature_steps: The number of temperature steps.
        """
        self.temperatures = np.linspace(
            temperature_min, temperature_max, temperature_steps
        )

    def set_volumes(
        self, volume_min: float, volume_max: float = None, volume_steps: int = 10
    ):
        """
        Set the volumes.

        Args:
            volume_min: The minimum volume.
            volume_max: The maximum volume.
            volume_steps: The number of volume steps.
        """
        if volume_max is None:
            volume_max = 1.1 * volume_min
        self.volumes = np.linspace(volume_min, volume_max, volume_steps)

    def meshgrid(self) -> np.ndarray:
        """
        Create a meshgrid of volumes and temperatures.

        Returns:
            The meshgrid of volumes and temperatures.
        """
        return np.meshgrid(self.volumes, self.temperatures)

    def get_minimum_energy_path(self, pressure: np.ndarray = None) -> np.ndarray:
        """
        Get the minimum energy path.

        Args:
            pressure: The pressure.

        Returns:
            The minimum energy path.
        """
        if pressure is not None:
            raise NotImplementedError()
        v_min_lst = []
        for c in self._coeff.T:
            v_min = np.roots(np.polyder(c, 1))
            p_der2 = np.polyder(c, 2)
            p_val2 = np.polyval(p_der2, v_min)
            v_m_lst = v_min[p_val2 > 0]
            if len(v_m_lst) > 0:
                v_min_lst.append(v_m_lst[0])
            else:
                v_min_lst.append(np.nan)
        return np.array(v_min_lst)

    def get_free_energy(
        self, vol: np.ndarray, pressure: np.ndarray = None
    ) -> np.ndarray:
        """
        Get the free energy.

        Args:
            vol: The volume.
            pressure: The pressure.

        Returns:
            The free energy.
        """
        if not pressure:
            return np.polyval(self._coeff, vol)
        else:
            raise NotImplementedError()

    def interpolate_volume(self, volumes: np.ndarray, fit_order: int = None):
        """
        Interpolate the volumes.

        Args:
            volumes: The volumes.
            fit_order: The order of the polynomial fit.

        Returns:
            The interpolated volume.
        """
        if fit_order is not None:
            self._fit_order = fit_order
        new = self.copy()
        new.volumes = volumes
        new.energies = np.array([np.polyval(self._coeff, v) for v in volumes]).T
        return new

    def _compute_thermo(self):
        """
        Compute the thermodynamic quantities.
        """
        self._entropy, self._pressure = np.gradient(
            -self.energies, self._d_temp, self._d_vol
        )

    def get_free_energy_p(self) -> np.ndarray:
        """
        Get the free energy at the minimum energy path.

        Returns:
            The free energy at the minimum energy path.
        """
        coeff = np.polyfit(self._volumes, self.energies.T, deg=self._fit_order)
        return np.polyval(coeff, self.get_minimum_energy_path())

    def get_entropy_p(self) -> np.ndarray:
        """
        Get the entropy at the minimum energy path.

        Returns:
            The entropy at the minimum energy path.
        """
        s_coeff = np.polyfit(self._volumes, self.entropy.T, deg=self._fit_order)
        return np.polyval(s_coeff, self.get_minimum_energy_path())

    def get_entropy_v(self) -> np.ndarray:
        """
        Get the entropy at constant volume.

        Returns:
            The entropy at constant volume.
        """
        eq_volume = self.volumes[0]
        s_coeff = np.polyfit(self.volumes, self.entropy.T, deg=self._fit_order)
        const_v = eq_volume * np.ones(len(s_coeff.T))
        return np.polyval(s_coeff, const_v)

    def plot_free_energy(self):
        """
        Plot the free energy.
        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        plt.plot(self.temperatures, self.get_free_energy_p() / self.num_atoms)
        plt.xlabel("Temperature [K]")
        plt.ylabel("Free energy [eV]")

    def plot_entropy(self):
        """
        Plot the entropy.
        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        plt.plot(
            self.temperatures,
            self.eV_to_J_per_mol / self.num_atoms * self.get_entropy_p(),
            label="S$_p$",
        )
        plt.plot(
            self.temperatures,
            self.eV_to_J_per_mol / self.num_atoms * self.get_entropy_v(),
            label="S$_V$",
        )
        plt.legend()
        plt.xlabel("Temperature [K]")
        plt.ylabel("Entropy [J K$^{-1}$ mol-atoms$^{-1}$]")

    def plot_heat_capacity(self, to_kB: bool = True):
        """
        Plot the heat capacity.

        Args:
            to_kB: Convert the heat capacity to kB units.
        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        if to_kB:
            units = self.kB / self.num_atoms
            plt.ylabel("Heat capacity [kB]")
        else:
            units = self.eV_to_J_per_mol
            plt.ylabel("Heat capacity [J K$^{-1}$ mol-atoms$^{-1}$]")
        temps = self.temperatures[:-2]
        c_p = temps * np.gradient(self.get_entropy_p(), self._d_temp)[:-2]
        c_v = temps * np.gradient(self.get_entropy_v(), self._d_temp)[:-2]
        plt.plot(temps, units * c_p, label="c$_p$")
        plt.plot(temps, units * c_v, label="c$_v$")
        plt.legend(loc="lower right")
        plt.xlabel("Temperature [K]")

    def contour_pressure(self) -> None:
        """
        Plot the contour of pressure.
        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        x, y = self.meshgrid()
        p_coeff = np.polyfit(self.volumes, self.pressure.T, deg=self._fit_order)
        p_grid = np.array([np.polyval(p_coeff, v) for v in self._volumes]).T
        plt.contourf(x, y, p_grid)
        plt.plot(self.get_minimum_energy_path(), self.temperatures)
        plt.xlabel("Volume [$\AA^3$]")
        plt.ylabel("Temperature [K]")

    def contour_entropy(self) -> None:
        """
        Plot the contour of entropy.
        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        s_coeff = np.polyfit(self.volumes, self.entropy.T, deg=self._fit_order)
        s_grid = np.array([np.polyval(s_coeff, v) for v in self.volumes]).T
        x, y = self.meshgrid()
        plt.contourf(x, y, s_grid)
        plt.plot(self.get_minimum_energy_path(), self.temperatures)
        plt.xlabel("Volume [$\AA^3$]")
        plt.ylabel("Temperature [K]")

    def plot_contourf(self, ax=None, show_min_erg_path=False):
        """
        Plot the contourf of energies.

        Args:
            ax: The matplotlib axes object.
            show_min_erg_path: Whether to show the minimum energy path.

        Returns:
            The matplotlib axes object.
        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        x, y = self.meshgrid()
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.contourf(x, y, self.energies)
        if show_min_erg_path:
            plt.plot(self.get_minimum_energy_path(), self.temperatures, "w--")
        plt.xlabel("Volume [$\AA^3$]")
        plt.ylabel("Temperature [K]")
        return ax

    def plot_min_energy_path(self, *args, ax=None, **qwargs):
        """
        Plot the minimum energy path.

        Args:
            *args: Additional arguments to pass to the plot function.
            ax: The matplotlib axes object.
            **qwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            The matplotlib axes object.
        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.xlabel("Volume [$\AA^3$]")
            ax.ylabel("Temperature [K]")
        ax.plot(self.get_minimum_energy_path(), self.temperatures, *args, **qwargs)
        return ax


def get_thermo_bulk_model(temperatures: np.ndarray, debye_model) -> ThermoBulk:
    """
    Get the thermo bulk model.

    Args:
        temperatures: The temperatures.
        debye_model: The Debye model.

    Returns:
        The thermo bulk model.
    """
    thermo = ThermoBulk()
    thermo.temperatures = temperatures
    thermo.volumes = debye_model.volume
    thermo.energies = (
        debye_model.interpolate()
        + debye_model.energy_vib(T=thermo.temperatures, low_T_limit=True).T
    )
    return thermo
