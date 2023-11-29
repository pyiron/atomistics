import numpy as np

from atomistics.shared.thermo.debye import get_debye_model
from atomistics.shared.thermo.thermo import get_thermo_bulk_model


def get_thermal_expansion_with_evcurve(
    fit_dict, masses, t_min=1, t_max=1500, t_step=50, temperatures=None
):
    if temperatures is None:
        temperatures = np.arange(t_min, t_max + t_step, t_step)
    debye_model = get_debye_model(fit_dict=fit_dict, masses=masses, num_steps=50)
    pes = get_thermo_bulk_model(
        temperatures=temperatures,
        debye_model=debye_model,
    )
    return temperatures, pes.get_minimum_energy_path()
