import numpy as np
import scipy.constants
import scipy.optimize
from ase.eos import (
    birch,
    murnaghan,
    pouriertarantola,
)
from ase.eos import (
    birchmurnaghan as birchmurnaghan_energy,
)
from ase.eos import (
    vinet as vinet_energy,
)

eV_div_A3_to_GPa = (
    1e21 / scipy.constants.physical_constants["joule-electron volt relationship"][0]
)


def fitfunction(
    parameters: tuple[float], vol: np.ndarray, fittype: str = "vinet"
) -> np.ndarray:
    """
    Fit the energy volume curve

    Args:
        parameters (list): [E0, B0, BP, V0] list of fit parameters
        vol (float/numpy.dnarray): single volume or a vector of volumes as numpy array
        fittype (str): on of the following ['birch', 'birchmurnaghan', 'murnaghan', 'pouriertarantola', 'vinet']

    Returns:
        (float/numpy.dnarray): single energy as float or a vector of energies as numpy array
    """
    [E0, b0, bp, V0] = parameters
    # Unit correction
    B0 = b0 / eV_div_A3_to_GPa
    BP = bp
    V = vol
    if fittype.lower() == "birchmurnaghan":
        return birchmurnaghan_energy(V, E0, B0, BP, V0)
    elif fittype.lower() == "vinet":
        return vinet_energy(V, E0, B0, BP, V0)
    elif fittype.lower() == "murnaghan":
        return murnaghan(V, E0, B0, BP, V0)
    elif fittype.lower() == "pouriertarantola":
        return pouriertarantola(V, E0, B0, BP, V0)
    elif fittype.lower() == "birch":
        return birch(V, E0, B0, BP, V0)
    else:
        raise ValueError


def interpolate_energy(fit_dict: dict, volumes: np.ndarray) -> np.ndarray:
    """
    Interpolate the energy values for given volumes using the fit_dict.

    Args:
        fit_dict (dict): Dictionary containing the fit results
        volumes (np.ndarray): Array of volumes for which to interpolate energy values

    Returns:
        np.ndarray: Array of interpolated energy values
    """
    if fit_dict["fit_dict"]["fit_type"] == "polynomial":
        return np.poly1d(fit_dict["fit_dict"]["poly_fit"])(volumes)
    elif fit_dict["fit_dict"]["fit_type"] in [
        "birch",
        "birchmurnaghan",
        "murnaghan",
        "pouriertarantola",
        "vinet",
    ]:
        parameters = [
            fit_dict["energy_eq"],
            fit_dict["bulkmodul_eq"],
            fit_dict["b_prime_eq"],
            fit_dict["volume_eq"],
        ]
        return fitfunction(
            parameters=parameters, vol=volumes, fittype=fit_dict["fit_dict"]["fit_type"]
        )
    else:
        raise ValueError("Unsupported fit_type: ", fit_dict["fit_dict"]["fit_type"])


def fit_leastsq(
    p0: tuple[float], datax: np.ndarray, datay: np.ndarray, fittype: str = "vinet"
):
    """
    Least square fit

    Args:
        p0 (list): [E0, B0, BP, V0] list of fit parameters
        datax (float/numpy.dnarray): volumes to fit
        datay (float/numpy.dnarray): energies corresponding to the volumes
        fittype (str): on of the following ['birch', 'birchmurnaghan', 'murnaghan', 'pouriertarantola', 'vinet']

    Returns:
        list: [E0, B0, BP, V0], [E0_err, B0_err, BP_err, V0_err]
    """
    # http://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i

    def errfunc(p, x, y, fittype):
        return fitfunction(p, x, fittype) - y

    pfit, pcov, infodict, errmsg, success = scipy.optimize.leastsq(
        errfunc, p0, args=(datax, datay, fittype), full_output=1, epsfcn=0.0001
    )

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay, fittype) ** 2).sum() / (
            len(datay) - len(p0)
        )
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except TypeError:
            error.append(0.00)
    pfit_leastsq = pfit
    perr_leastsq = np.array(error)
    return pfit_leastsq, perr_leastsq


def fit_leastsq_eos(
    volume_lst: np.ndarray, energy_lst: np.ndarray, fittype: str = "birchmurnaghan"
):
    """
    Internal helper function for the least square fit

    Args:
        volume_lst (list/numpy.dnarray/None): vector of volumes
        energy_lst (list/numpy.dnarray/None): vector of energies
        fittype (str): on of the following ['birch', 'birchmurnaghan', 'murnaghan', 'pouriertarantola', 'vinet']

    Returns:
        list: [E0, B0, BP, V0], [E0_err, B0_err, BP_err, V0_err]
    """
    vol_lst = np.array(volume_lst).flatten()
    eng_lst = np.array(energy_lst).flatten()
    a, b, c = np.polyfit(vol_lst, eng_lst, 2)
    v0 = -b / (2 * a)
    pfit_leastsq, perr_leastsq = fit_leastsq(
        [a * v0**2 + b * v0 + c, 2 * a * v0 * eV_div_A3_to_GPa, 4, v0],
        vol_lst,
        eng_lst,
        fittype,
    )
    return pfit_leastsq, perr_leastsq  # [e0, b0, bP, v0]


def get_error(x_lst: np.ndarray, y_lst: np.ndarray, p_fit) -> float:
    """
    Calculate the mean squared error between the observed and predicted values.

    Args:
        x_lst (np.ndarray): Array of x values
        y_lst (np.ndarray): Array of observed y values
        p_fit (np.poly1d): Polynomial fit function

    Returns:
        float: Mean squared error
    """
    y_fit_lst = np.array(p_fit(x_lst))
    error_lst = (y_lst - y_fit_lst) ** 2
    return np.mean(error_lst)


def fit_equation_of_state(
    volume_lst: np.ndarray, energy_lst: np.ndarray, fittype: str
) -> dict:
    """
    Fit the equation of state to the given volume and energy data.

    Args:
        volume_lst (np.ndarray): Array of volumes
        energy_lst (np.ndarray): Array of energies
        fittype (str): Type of fit to perform

    Returns:
        dict: Dictionary containing the fit results
    """
    fit_dict = {}
    pfit_leastsq, perr_leastsq = fit_leastsq_eos(
        volume_lst=volume_lst, energy_lst=energy_lst, fittype=fittype
    )
    fit_dict["fit_type"] = fittype
    fit_dict["volume_eq"] = pfit_leastsq[3]
    fit_dict["energy_eq"] = pfit_leastsq[0]
    fit_dict["bulkmodul_eq"] = pfit_leastsq[1]
    fit_dict["b_prime_eq"] = pfit_leastsq[2]
    fit_dict["least_square_error"] = perr_leastsq  # [e0, b0, bP, v0]
    fit_dict["volume"] = volume_lst
    fit_dict["energy"] = energy_lst

    return fit_dict


def fit_polynomial(
    volume_lst: np.ndarray, energy_lst: np.ndarray, fit_order: int
) -> dict:
    """
    Fit a polynomial to the given volume and energy data.

    Args:
        volume_lst (np.ndarray): Array of volumes
        energy_lst (np.ndarray): Array of energies
        fit_order (int): Order of the polynomial fit

    Returns:
        dict: Dictionary containing the fit results
    """
    fit_dict = {}

    # compute a polynomial fit
    z = np.polyfit(volume_lst, energy_lst, fit_order)
    p_fit = np.poly1d(z)
    fit_dict["poly_fit"] = z

    # get equilibrium lattice constant
    # search for the local minimum with the lowest energy
    p_deriv_1 = np.polyder(p_fit, 1)
    roots = np.roots(p_deriv_1)

    volume_eq_lst = np.array(
        [
            np.real(r)
            for r in roots
            if (
                abs(np.imag(r)) < 1e-10
                and r >= min(volume_lst)
                and r <= max(volume_lst)
            )
        ]
    )

    e_eq_lst = p_fit(volume_eq_lst)
    arg = np.argsort(e_eq_lst)
    if len(e_eq_lst) == 0:
        return None
    e_eq = e_eq_lst[arg][0]
    volume_eq = volume_eq_lst[arg][0]

    # get bulk modulus at equ. lattice const.
    p_2deriv = np.polyder(p_fit, 2)
    p_3deriv = np.polyder(p_fit, 3)
    a2 = p_2deriv(volume_eq)
    a3 = p_3deriv(volume_eq)

    b_prime = -(volume_eq * a3 / a2 + 1)

    fit_dict["fit_type"] = "polynomial"
    fit_dict["fit_order"] = fit_order
    fit_dict["volume_eq"] = volume_eq
    fit_dict["energy_eq"] = e_eq
    fit_dict["bulkmodul_eq"] = eV_div_A3_to_GPa * volume_eq * a2
    fit_dict["b_prime_eq"] = b_prime
    fit_dict["least_square_error"] = get_error(volume_lst, energy_lst, p_fit)
    fit_dict["volume"] = volume_lst
    fit_dict["energy"] = energy_lst
    return fit_dict


class EnergyVolumeFit:
    """
    Fit energy volume curves

    Args:
        volume_lst (list/numpy.dnarray): vector of volumes
        energy_lst (list/numpy.dnarray): vector of energies

    Attributes:

        .. attribute:: volume_lst

            vector of volumes

        .. attribute:: energy_lst

            vector of energies

        .. attribute:: fit_dict

            dictionary of fit parameters
    """

    def __init__(self, volume_lst: np.ndarray = None, energy_lst: np.ndarray = None):
        """
        Initialize the EnergyVolumeFit object.

        Args:
            volume_lst (np.ndarray, optional): Vector of volumes. Defaults to None.
            energy_lst (np.ndarray, optional): Vector of energies. Defaults to None.
        """
        self._volume_lst = volume_lst
        self._energy_lst = energy_lst
        self._fit_dict = None

    @property
    def volume_lst(self) -> np.ndarray:
        """
        Get the vector of volumes.

        Returns:
            np.ndarray: Vector of volumes.
        """
        return self._volume_lst

    @volume_lst.setter
    def volume_lst(self, vol_lst: np.ndarray):
        """
        Set the vector of volumes.

        Args:
            vol_lst (np.ndarray): Vector of volumes.
        """
        self._volume_lst = vol_lst

    @property
    def energy_lst(self) -> np.ndarray:
        """
        Get the vector of energies.

        Returns:
            np.ndarray: Vector of energies.
        """
        return self._energy_lst

    @energy_lst.setter
    def energy_lst(self, eng_lst: np.ndarray):
        """
        Set the vector of energies.

        Args:
            eng_lst (np.ndarray): Vector of energies.
        """
        self._energy_lst = eng_lst

    @property
    def fit_dict(self) -> dict:
        """
        Get the fit dictionary.

        Returns:
            dict: Fit dictionary.
        """
        return self._fit_dict

    def _get_volume_and_energy_lst(
        self, volume_lst: np.ndarray = None, energy_lst: np.ndarray = None
    ) -> tuple[np.ndarray]:
        """
        Internal function to get the vector of volumes and the vector of energies

        Args:
            volume_lst (list/numpy.dnarray/None): vector of volumes
            energy_lst (list/numpy.dnarray/None): vector of energies

        Returns:
            list: vector of volumes and vector of energies
        """
        if volume_lst is None:
            if self._volume_lst is None:
                raise ValueError("Volume list not set.")
            volume_lst = self._volume_lst
        if energy_lst is None:
            if self._energy_lst is None:
                raise ValueError("Volume list not set.")
            energy_lst = self._energy_lst
        return volume_lst, energy_lst

    def fit(self, fit_type: str = "polynomial", fit_order: int = 3) -> dict:
        """
        Fit the energy volume curves.

        Args:
            fit_type (str, optional): Type of fit to perform. Defaults to "polynomial".
            fit_order (int, optional): Order of the polynomial fit. Defaults to 3.

        Returns:
            dict: Dictionary containing the fit results.
        """
        if fit_type == "polynomial":
            self._fit_dict = self.fit_polynomial(fit_order=fit_order)
        elif fit_type in [
            "birch",
            "birchmurnaghan",
            "murnaghan",
            "pouriertarantola",
            "vinet",
        ]:
            self._fit_dict = self.fit_eos_general(fittype=fit_type)
        else:
            raise ValueError(
                "fit_type is unrecognized, the supported fit types are ['polynomial', "
                "'birch', 'birchmurnaghan', 'murnaghan', 'pouriertarantola', 'vinet'] "
                + str(fit_type)
                + " is not a supported fit_type"
            )
        return self._fit_dict

    def fit_eos_general(
        self,
        volume_lst: np.ndarray = None,
        energy_lst: np.ndarray = None,
        fittype: str = "birchmurnaghan",
    ) -> dict:
        """
        Fit one of the equations of state.

        Args:
            volume_lst (np.ndarray, optional): Vector of volumes. Defaults to None.
            energy_lst (np.ndarray, optional): Vector of energies. Defaults to None.
            fittype (str, optional): Type of fit to perform. Defaults to "birchmurnaghan".

        Returns:
            dict: Dictionary containing the fit results.
        """
        volume_lst, energy_lst = self._get_volume_and_energy_lst(
            volume_lst=volume_lst, energy_lst=energy_lst
        )
        return fit_equation_of_state(
            volume_lst=volume_lst, energy_lst=energy_lst, fittype=fittype
        )

    def fit_polynomial(
        self,
        volume_lst: np.ndarray = None,
        energy_lst: np.ndarray = None,
        fit_order: int = 3,
    ) -> dict:
        """
        Fit a polynomial.

        Args:
            volume_lst (np.ndarray, optional): Vector of volumes. Defaults to None.
            energy_lst (np.ndarray, optional): Vector of energies. Defaults to None.
            fit_order (int, optional): Order of the polynomial fit. Defaults to 3.

        Returns:
            dict: Dictionary containing the fit results.
        """
        volume_lst, energy_lst = self._get_volume_and_energy_lst(
            volume_lst=volume_lst, energy_lst=energy_lst
        )
        return fit_polynomial(
            volume_lst=volume_lst, energy_lst=energy_lst, fit_order=fit_order
        )

    def interpolate_energy(self, volume_lst: np.ndarray) -> np.ndarray:
        """
        Interpolate the energy values for the corresponding energy volume fit defined in the fit dictionary.

        Args:
            volume_lst (np.ndarray): List of volumes.

        Returns:
            np.ndarray: List of energies.
        """
        if not self._fit_dict:
            return ValueError("parameter 'fit_dict' has to be defined!")
        return interpolate_energy(fit_dict=self.fit_dict, volumes=volume_lst)

    @staticmethod
    def birchmurnaghan_energy(
        V: np.ndarray, E0: float, B0: float, BP: float, V0: float
    ) -> np.ndarray:
        """
        BirchMurnaghan equation from PRB 70, 224107

        Args:
            V (np.ndarray): Vector of volumes.
            E0 (float): Energy at equilibrium volume.
            B0 (float): Bulk modulus at equilibrium volume.
            BP (float): Pressure derivative of bulk modulus at equilibrium volume.
            V0 (float): Equilibrium volume.

        Returns:
            np.ndarray: Vector of energies.
        """
        return birchmurnaghan_energy(V, E0, B0, BP, V0)

    @staticmethod
    def vinet_energy(
        V: np.ndarray, E0: float, B0: float, BP: float, V0: float
    ) -> np.ndarray:
        """
        Vinet equation from PRB 70, 224107

        Args:
            V (np.ndarray): Vector of volumes.
            E0 (float): Energy at equilibrium volume.
            B0 (float): Bulk modulus at equilibrium volume.
            BP (float): Pressure derivative of bulk modulus at equilibrium volume.
            V0 (float): Equilibrium volume.

        Returns:
            np.ndarray: Vector of energies.
        """
        return vinet_energy(V, E0, B0, BP, V0)

    @staticmethod
    def murnaghan(
        V: np.ndarray, E0: float, B0: float, BP: float, V0: float
    ) -> np.ndarray:
        """
        Murnaghan equation from PRB 28,5480 (1983)

        Args:
            V (np.ndarray): Vector of volumes.
            E0 (float): Energy at equilibrium volume.
            B0 (float): Bulk modulus at equilibrium volume.
            BP (float): Pressure derivative of bulk modulus at equilibrium volume.
            V0 (float): Equilibrium volume.

        Returns:
            np.ndarray: Vector of energies.
        """
        return murnaghan(V, E0, B0, BP, V0)

    @staticmethod
    def birch(V: np.ndarray, E0: float, B0: float, BP: float, V0: float) -> np.ndarray:
        """
        Birch equation from Intermetallic compounds: Principles and Practice, Vol. I: Principles
        Chapter 9 pages 195-210 by M. Mehl. B. Klein, D. Papaconstantopoulos
        paper downloaded from Web

        case where n=0

        Args:
            V (np.ndarray): Vector of volumes.
            E0 (float): Energy at equilibrium volume.
            B0 (float): Bulk modulus at equilibrium volume.
            BP (float): Pressure derivative of bulk modulus at equilibrium volume.
            V0 (float): Equilibrium volume.

        Returns:
            np.ndarray: Vector of energies.
        """
        return birch(V, E0, B0, BP, V0)

    @staticmethod
    def pouriertarantola(
        V: np.ndarray, E0: float, B0: float, BP: float, V0: float
    ) -> np.ndarray:
        """
        Pouriertarantola equation

        Args:
            V (np.ndarray): Vector of volumes.
            E0 (float): Energy at equilibrium volume.
            B0 (float): Bulk modulus at equilibrium volume.
            BP (float): Pressure derivative of bulk modulus at equilibrium volume.
            V0 (float): Equilibrium volume.

        Returns:
            np.ndarray: Vector of energies.
        """
        return pouriertarantola(V, E0, B0, BP, V0)


def get_energy_volume_curve_fit(
    volume_lst: np.ndarray = None, energy_lst: np.ndarray = None
) -> EnergyVolumeFit:
    """
    Create an instance of EnergyVolumeFit class with the given volume and energy lists.

    Args:
        volume_lst (np.ndarray, optional): Vector of volumes. Defaults to None.
        energy_lst (np.ndarray, optional): Vector of energies. Defaults to None.

    Returns:
        EnergyVolumeFit: Instance of EnergyVolumeFit class.
    """
    return EnergyVolumeFit(volume_lst=volume_lst, energy_lst=energy_lst)
