import numpy as np
import scipy

from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.calculators.hessian import get_displacement


kb = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0] * 1000
hartree = scipy.constants.physical_constants["atomic unit of energy"][0]  # J
bohr_r = scipy.constants.physical_constants["Bohr radius"][0]  # in m
hbar_str = "Planck constant over 2 pi in eV s"
hbar = scipy.constants.physical_constants[hbar_str][0]  # eV s
u = scipy.constants.physical_constants["atomic mass constant"][0]  # in kg


def get_free_energy_classical(structure, force_constant, temperature):
    frequency_dict = get_einstein_frequencies(
        structure=structure, force_constant=force_constant
    )

    def get_free_energy(frequency, temperature):
        return kb * temperature * np.log(frequency / (kb * temperature))

    return {
        el: get_free_energy(frequency=frequency, temperature=temperature)
        for el, frequency in frequency_dict.items()
    }


def get_free_energy_quantum_mechanical(structure, force_constant, temperature):
    frequency_dict = get_einstein_frequencies(
        structure=structure, force_constant=force_constant
    )

    def get_free_energy(frequency, temperature):
        return frequency / 2 + kb * temperature * np.log(
            1 - np.exp(-(frequency / (kb * temperature)))
        )

    return {
        el: get_free_energy(frequency=frequency, temperature=temperature)
        for el, frequency in frequency_dict.items()
    }


def get_einstein_frequencies(structure, force_constant):
    return {
        el: 1000
        * np.sqrt(hbar * hbar * force_constant * hartree / u / mass / bohr_r / bohr_r)
        for el, mass in zip(structure.get_chemical_symbols(), structure.get_masses())
    }


def get_energy_pot(structure, structure_equilibrium, force_constant):
    dis = get_displacement(
        structure_equilibrium=structure_equilibrium, structure=structure
    )
    return sum(
        0.5 * force_constant * hartree / bohr_r / bohr_r * np.linalg.norm(dis, axis=1)
    )


def get_forces(structure, structure_equilibrium, force_constant):
    dis = get_displacement(
        structure_equilibrium=structure_equilibrium, structure=structure
    )
    return dis * force_constant * hartree / bohr_r / bohr_r


@as_task_dict_evaluator
def evaluate_with_einstein_model(
    structure,
    structure_equilibrium,
    force_constant,
    tasks,
):
    results = {}
    if "calc_energy" in tasks or "calc_forces" in tasks:
        if "calc_energy" in tasks:
            results["energy"] = get_energy_pot(
                structure=structure,
                structure_equilibrium=structure_equilibrium,
                force_constant=force_constant,
            )
        if "calc_forces" in tasks:
            results["forces"] = get_forces(
                structure=structure,
                structure_equilibrium=structure_equilibrium,
                force_constant=force_constant,
            )
    else:
        raise ValueError("The ASE calculator does not implement:", tasks)
    return results
