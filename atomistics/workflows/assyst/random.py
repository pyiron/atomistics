from random import choices

import numpy as np
from tqdm.auto import tqdm


def shake(displacement=0.1):
    """
    Return a function that randomly displaces atoms in structures.

    Args:
        displacement (float): standard deviation of atomic displacement
    """

    def mod(structure):
        structure.positions += np.random.normal(
            scale=displacement, size=structure.positions.shape
        )
        return structure

    return mod


def stretch(hydro: float = 0.05, shear: float = 0.005):
    """
    Return a function that strains structures.

    Random strains are drawn from a uniform distribution within the positive
    and negative limits given.

    Args:
        hydro (float): Maximum strain along normal axes
        shear (float): Maximum strain along shear axes
    """

    def mod(structure):
        E = shear * (2 * np.random.rand(3, 3) - 1)
        E = 0.5 * (E + E.T)  # symmetrize
        np.fill_diagonal(E, hydro * (2 * np.random.rand(3) - 1))
        structure.apply_strain(E)
        return structure

    return mod


def fill_container(
    source: list[dict],
    repetitions: int = 4,
    combine: int = 1,
    modifiers=((0.5, shake()), (0.5, stretch())),
    filterf=None,
):
    """
    Fill a container with new structures.

    Iterates over all structures in `source`
    """
    ps, mods = zip(*modifiers)
    sink = []
    for entry_dict in tqdm(source, total=len(source)):
        structure = entry_dict["structure"]
        for _ in range(repetitions):
            for i in range(10):
                s = structure.copy()
                for mod in choices(mods, weights=ps, k=combine):
                    s = mod(s)
                if filterf is None or filterf(s):
                    sink.append({
                        "structure": s,
                        "identifier": entry_dict["identifier"],
                        "symmetry": entry_dict["symmetry"],
                        "repeat": entry_dict["repeat"],
                    })
                    break
            else:
                print(
                    "WARN: Tried 10 times to find a structures, but "
                    "filter is never satisfied."
                )
    return sink


def rattle(
    source: list[dict],
    rattle_disp,
    rattle_strain,
    rattle_repetitions,
    stretch_hydro,
    stretch_shear,
    stretch_repetitions,
    filterf,
):
    result = fill_container(
        source=[entry for entry in source if len(entry["structure"]) > 1],
        repetitions=rattle_repetitions,
        combine=2,
        modifiers=((1, shake(rattle_disp)), (1, stretch(rattle_strain))),
        filterf=filterf,
    )
    result += fill_container(
        source=source,
        repetitions=stretch_repetitions,
        combine=1,
        modifiers=(
            (0.7, stretch(hydro=stretch_hydro, shear=0.05)),
            (0.3, stretch(hydro=0.05, shear=stretch_shear)),
        ),
        filterf=filterf,
    )
    return result