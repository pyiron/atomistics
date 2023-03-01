import random
import warnings
import itertools
from multiprocessing import cpu_count
from ase.data import atomic_numbers
from ase.atoms import Atoms
import numpy as np
from sqsgenerator import sqs_optimize
from typing import Dict, Optional, Union, Iterable


def chemical_formula(atoms: Atoms) -> str:
    def group_symbols():
        for species, same in itertools.groupby(atoms.get_chemical_symbols()):
            num_same = len(list(same))
            yield species if num_same == 1 else f"{species}{num_same}"

    return "".join(group_symbols())


def map_dict(f, d: Dict) -> Dict:
    return {k: f(v) for k, v in d.items()}


def mole_fractions_to_composition(
    mole_fractions: Dict[str, float], num_atoms: int
) -> Dict[str, int]:
    # if the sum of x is less than 1 - 1/n then we are missing at least one atoms
    if not (1.0 - 1 / num_atoms) < sum(mole_fractions.values()) < (1.0 + 1 / num_atoms):
        raise ValueError(
            "mole-fractions must sum up to one: {}".format(sum(mole_fractions.values()))
        )

    composition = map_dict(lambda x: x * num_atoms, mole_fractions)
    # check to avoid partial occupation -> x_i * num_atoms is not an integer number
    if any(
        map(
            lambda occupation: not float.is_integer(round(occupation, 1)),
            composition.values(),
        )
    ):
        # at least one of the specified species exhibits fractional occupation, we try to fix it by rounding
        composition_ = map_dict(lambda occ: int(round(occ)), composition)
        warnings.warn(
            "The current mole-fraction specification cannot be applied to {} atoms, "
            "as it would lead to fractional occupation. Hence, I have changed it from "
            '"{}" -> "{}"'.format(
                num_atoms,
                mole_fractions,
                map_dict(lambda n: n / num_atoms, composition_),
            )
        )
        composition = composition_

    # due to rounding errors there might be a difference of one atom
    actual_atoms = sum(composition.values())
    diff = actual_atoms - num_atoms
    if abs(diff) == 1:
        # it is not possible to distribute atoms equally e.g x_a = x_b = x_c = 1/3 on 32 atoms
        # we remove one randomly bet we inform the user
        removed_species = random.choice(tuple(composition))
        composition[removed_species] -= 1
        warnings.warn(
            'It is not possible to distribute the species properly. Therefore one "{}" atom was removed. '
            "This changes the input mole-fraction specification. "
            '"{}" -> "{}"'.format(
                removed_species,
                mole_fractions,
                map_dict(lambda n: n / num_atoms, composition),
            )
        )
    elif abs(diff) > 1:
        # something else is wrong with the mole-fractions input
        raise ValueError(
            "Cannot interpret mole-fraction dict {}".format(mole_fractions)
        )

    return composition


def remap_sro(species: Iterable[str], array: np.ndarray):
    # remaps computed short-range order parameters to style of sqsgenerator=v0.0.5
    species = tuple(sorted(species, key=lambda abbr: atomic_numbers[abbr]))
    return {
        "{}-{}".format(si, sj): array[:, i, j].tolist()
        for (i, si), (j, sj) in itertools.product(
            enumerate(species), enumerate(species)
        )
        if j >= i
    }


def remap_sqs_results(result):
    # makes new interface compatible with old one
    return result["structure"], remap_sro(
        set(result["structure"].get_chemical_symbols()), result["parameters"]
    )


def transpose(it):
    return zip(*it)


def get_sqs_structures(
    structure: Atoms,
    mole_fractions: Dict[str, Union[float, int]],
    weights: Optional[Dict[int, float]] = None,
    objective: Union[float, np.ndarray] = 0.0,
    iterations: Union[float, int] = 1e6,
    output_structures: int = 10,
    mode: str = "random",
    num_threads: Optional[int] = None,
    prefactors: Optional[Union[float, np.ndarray]] = None,
    pair_weights: Optional[np.ndarray] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    which: Optional[Iterable[int]] = None,
    shell_distances: Optional[Iterable[int]] = None,
    minimal: Optional[bool] = True,
    similar: Optional[bool] = True,
):
    composition = mole_fractions_to_composition(mole_fractions, len(structure))

    settings = dict(
        atol=atol,
        rtol=rtol,
        mode=mode,
        which=which,
        structure=structure,
        prefactors=prefactors,
        shell_weights=weights,
        iterations=iterations,
        composition=composition,
        pair_weights=pair_weights,
        target_objective=objective,
        shell_distances=shell_distances,
        threads_per_rank=num_threads or cpu_count(),
        max_output_configurations=output_structures,
    )
    # not specifying a parameter in settings causes sqsgenerator to choose a "sensible" default,
    # hence we remove all entries with a None value

    results, timings = sqs_optimize(
        {param: value for param, value in settings.items() if value is not None},
        minimal=minimal,
        similar=similar,
        make_structures=True,
        structure_format="ase",
    )

    structures, sro_breakdown = transpose(map(remap_sqs_results, results.values()))
    num_iterations = iterations
    cycle_time = np.average(list(map_dict(np.average, timings).values()))
    return structures, sro_breakdown, num_iterations, cycle_time
