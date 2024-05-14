import numpy as np
import scipy.constants
import scipy.optimize


eV_div_A3_to_GPa = (
    1e21 / scipy.constants.physical_constants["joule-electron volt relationship"][0]
)


# https://gitlab.com/ase/ase/blob/master/ase/eos.py
def birchmurnaghan_energy(
    V: np.ndarray, E0: float, B0: float, BP: float, V0: float
) -> np.ndarray:
    "BirchMurnaghan equation from PRB 70, 224107"
    eta = (V0 / V) ** (1 / 3)
    return E0 + 9 * B0 * V0 / 16 * (eta**2 - 1) ** 2 * (
        6 + BP * (eta**2 - 1) - 4 * eta**2
    )


def vinet_energy(
    V: np.ndarray, E0: float, B0: float, BP: float, V0: float
) -> np.ndarray:
    "Vinet equation from PRB 70, 224107"
    eta = (V / V0) ** (1 / 3)
    return E0 + 2 * B0 * V0 / (BP - 1) ** 2 * (
        2 - (5 + 3 * BP * (eta - 1) - 3 * eta) * np.exp(-3 * (BP - 1) * (eta - 1) / 2)
    )


def murnaghan(V: np.ndarray, E0: float, B0: float, BP: float, V0: float) -> np.ndarray:
    "From PRB 28,5480 (1983"
    E = E0 + B0 * V / BP * (((V0 / V) ** BP) / (BP - 1) + 1) - V0 * B0 / (BP - 1)
    return E


def birch(V: np.ndarray, E0: float, B0: float, BP: float, V0: float) -> np.ndarray:
    """
    From Intermetallic compounds: Principles and Practice, Vol. I: Principles
    Chapter 9 pages 195-210 by M. Mehl. B. Klein, D. Papaconstantopoulos
    paper downloaded from Web

    case where n=0
    """
    E = (
        E0
        + 9 / 8 * B0 * V0 * ((V0 / V) ** (2 / 3) - 1) ** 2
        + 9 / 16 * B0 * V0 * (BP - 4) * ((V0 / V) ** (2 / 3) - 1) ** 3
    )
    return E


def pouriertarantola(
    V: np.ndarray, E0: float, B0: float, BP: float, V0: float
) -> np.ndarray:
    "Pourier-Tarantola equation from PRB 70, 224107"
    eta = (V / V0) ** (1 / 3)
    squiggle = -3 * np.log(eta)

    E = E0 + B0 * V0 * squiggle**2 / 6 * (3 + squiggle * (BP - 2))
    return E


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

    Args:
        x_lst:
        y_lst:
        p_fit:

    Returns:
        numpy.dnarray
    """
    y_fit_lst = np.array(p_fit(x_lst))
    error_lst = (y_lst - y_fit_lst) ** 2
    return np.mean(error_lst)


def fit_equation_of_state(
    volume_lst: np.ndarray, energy_lst: np.ndarray, fittype: str
) -> dict:
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
    fit_dict = {}

    # compute a polynomial fit
    z = np.polyfit(volume_lst, energy_lst, fit_order)
    p_fit = np.poly1d(z)
    fit_dict["poly_fit"] = z

    # get equilibrium lattice constant
    # search for the local minimum with the lowest energy
    p_deriv_1 = np.polyder(p_fit, 1)
    roots = np.roots(p_deriv_1)

    # volume_eq_lst = np.array([np.real(r) for r in roots if np.abs(np.imag(r)) < 1e-10])
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
    # print ("v_eq:", arg, e_eq_lst)
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


class EnergyVolumeFit(object):
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
        self._volume_lst = volume_lst
        self._energy_lst = energy_lst
        self._fit_dict = None

    @property
    def volume_lst(self) -> np.ndarray:
        return self._volume_lst

    @volume_lst.setter
    def volume_lst(self, vol_lst: np.ndarray):
        self._volume_lst = vol_lst

    @property
    def energy_lst(self) -> np.ndarray:
        return self._energy_lst

    @energy_lst.setter
    def energy_lst(self, eng_lst: np.ndarray):
        self._energy_lst = eng_lst

    @property
    def fit_dict(self) -> dict:
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
    ):
        """
        Fit on of the equations of state

        Args:
            volume_lst (list/numpy.dnarray/None): vector of volumes
            energy_lst (list/numpy.dnarray/None): vector of energies
            fittype (str): on of the following ['birch', 'birchmurnaghan', 'murnaghan', 'pouriertarantola', 'vinet']

        Returns:
            dict: dictionary with fit results
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
    ):
        """
        Fit a polynomial

        Args:
            volume_lst (list/numpy.dnarray/None): vector of volumes
            energy_lst (list/numpy.dnarray/None): vector of energies
            fit_order (int): Degree of the polynomial

        Returns:
            dict: dictionary with fit results
        """
        volume_lst, energy_lst = self._get_volume_and_energy_lst(
            volume_lst=volume_lst, energy_lst=energy_lst
        )
        return fit_polynomial(
            volume_lst=volume_lst, energy_lst=energy_lst, fit_order=fit_order
        )

    def interpolate_energy(self, volume_lst: np.ndarray) -> np.ndarray:
        """
        Gives the energy value for the corresponding energy volume fit defined in the fit dictionary.

        Args:
            volume_lst: list of volumes

        Returns:
            list of energies

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
        """
        return birchmurnaghan_energy(V, E0, B0, BP, V0)

    @staticmethod
    def vinet_energy(
        V: np.ndarray, E0: float, B0: float, BP: float, V0: float
    ) -> np.ndarray:
        """
        Vinet equation from PRB 70, 224107
        """
        return vinet_energy(V, E0, B0, BP, V0)

    @staticmethod
    def murnaghan(
        V: np.ndarray, E0: float, B0: float, BP: float, V0: float
    ) -> np.ndarray:
        """
        From PRB 28,5480 (1983)
        """
        return murnaghan(V, E0, B0, BP, V0)

    @staticmethod
    def birch(V: np.ndarray, E0: float, B0: float, BP: float, V0: float) -> np.ndarray:
        """
        From Intermetallic compounds: Principles and Practice, Vol. I: Principles
        Chapter 9 pages 195-210 by M. Mehl. B. Klein, D. Papaconstantopoulos
        paper downloaded from Web

        case where n=0
        """
        return birch(V, E0, B0, BP, V0)

    @staticmethod
    def pouriertarantola(
        V: np.ndarray, E0: float, B0: float, BP: float, V0: float
    ) -> np.ndarray:
        return pouriertarantola(V, E0, B0, BP, V0)


def get_energy_volume_curve_fit(
    volume_lst: np.ndarray = None, energy_lst: np.ndarray = None
):
    return EnergyVolumeFit(volume_lst=volume_lst, energy_lst=energy_lst)
