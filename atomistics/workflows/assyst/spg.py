from typing import Union, List, Tuple
from itertools import product
import warnings

from ase.data import atomic_numbers
from tqdm.auto import tqdm
from structuretoolkit.analyse import get_symmetry
from structuretoolkit.common import center_coordinates_in_unit_cell
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError, VolumeError
from pyxtal.tolerance import Tol_matrix

from atomistics.workflows.assyst.util import RCORE, DistanceFilter


def _pyxtal(
    group: Union[int, List[int]],
    species: Tuple[str],
    num_ions: Tuple[int],
    dim=3,
    repeat=1,
    allow_exceptions=True,
    checker=lambda _: True,
    **kwargs,
) -> list[dict]:
    """
    Generate random crystal structures with PyXtal.

    `group` must be between 1 and the largest possible value for the given dimensionality:
        dim=3 => 1 - 230 (space groups)
        dim=2 => 1 -  80 (layer groups)
        dim=1 => 1 -  75 (rod groups)
        dim=0 => 1 -  58 (point groups)

    When `group` is passed as a list of integers or `repeat>1`, generate multiple structures and return them in a :class:`.StructureStorage`.

    Args:
        group (list of int, or int): the symmetry group to generate or a list of them
        species (tuple of str): which species to include, defines the stoichiometry together with `num_ions`
        num_ions (tuple of int): how many of each species to include, defines the stoichiometry together with `species`
        dim (int): dimensionality of the symmetry group, 0 is point groups, 1 is rod groups, 2 is layer groups and 3 is space groups
        repeat (int): how many random structures to generate
        storage (:class:`.StructureStorage`, optional): when generating multiple structures, add them to this instead of creating a new storage
        allow_exceptions (bool): when generating multiple structures, silence errors when the requested stoichiometry and symmetry group are incompatible
        **kwargs: passed to `pyxtal.pyxtal` function verbatim

    Returns:
        :class:`~.Atoms`: the generated structure, if repeat==1 and only one symmetry group is requested
        :class:`.StructureStorage`: a storage of all generated structure, if repeat>1 or multiple symmetry groups are requested

    Raises:
        ValueError: if stoichiometry and symmetry group are incompatible and allow_exceptions==False or only one structure is requested
    """
    def generate(group):
        s = pyxtal()
        factor = 1
        for _ in range(5):
            try:
                s.from_random(
                    dim=dim, group=group, species=species, numIons=num_ions, **kwargs
                )
                s = center_coordinates_in_unit_cell(s.to_ase())
                return s
            except RuntimeError as err:
                if err.args[0] == "long time to generate structure, check inputs":
                    warnings.warn(
                        f"pyxtal complained: {err.args} {factor} {dim} {group} {species} {num_ions}"
                    )
                if not err.args[0].startswith("Volume"):
                    raise
            except VolumeError:
                pass
            except:
                raise
            factor *= 1.5
        else:
            raise RuntimeError(
                "Failed to generate structure, aborted after factor: {factor}!"
            )

    # return a single structure
    stoich = "".join(f"{s}{n}" for s, n in zip(species, num_ions))
    if repeat == 1 and isinstance(group, int):
        return [{"structure": generate(group), "identifier": f"{stoich}_{group}_{0}", "symmetry": group, "repeat": repeat}]
    else:
        storage = []
        if isinstance(group, int):
            group = [group]
        failed_groups = []
        for g in tqdm(group, desc="Spacegroups"):
            for i in range(repeat):
                try:
                    for _ in range(5):
                        s = generate(g)
                        if checker(s):
                            break
                    else:
                        warnings.warn("Check failed 5 times in a row, skipping!")
                        continue
                except (Comp_CompatibilityError, RuntimeError) as e:
                    if allow_exceptions:
                        # This exception indicates that the stoichiometry is generally incompatible with the chosen group
                        # so we can just skip it
                        failed_groups.append(g)
                        break
                    else:
                        raise ValueError(
                            f"Symmetry group {g} incompatible with stoichiometry {stoich}!"
                        ) from None
                # some structures come out with really weird cell shapes, especially with low number of atoms
                # get the primitive cell as per spglib to "normalize" that a bit
                # at the same time we do *not* want to reduce the size of the cells, because having a few larger super
                # cells will allow us to sample their displacements a bit more
                ps = get_symmetry(structure=s).get_primitive_cell()
                if len(ps) == len(s):
                    s = ps
                storage.append(
                    {"structure": s, "identifier":f"{stoich}_{g}_{i}", "symmetry":g, "repeat":i}
                )
        if len(failed_groups) > 0:
            stoich = "".join(f"{s}{n}" for s, n in zip(species, num_ions))
            warnings.warn(
                f'Groups [{", ".join(map(str,failed_groups))}] could not be generated with stoichiometry {stoich}!'
            )
        return storage


def spg(
    elements,
    max_atoms,
    stoichiometry,
    min_dist=None,
):
    if min_dist is not None:
        if isinstance(min_dist, float):
            tm = Tol_matrix.from_single_value(min_dist)
        elif isinstance(min_dist, dict):
            tm = Tol_matrix(
                *(
                    (
                        atomic_numbers[e1],
                        atomic_numbers[e2],
                        min_dist[e1] + min_dist[e2],
                    )
                    for e1, e2 in product(min_dist, repeat=2)
                )
            )
        else:
            assert False, f"min_dist cannot by of type {type(min_dist)}: {min_dist}!"
    else:
        # function is called radii, but source code suggest it is actually used
        # to check the *distance* between to atom pairs, so we multiply by two
        # here (because the pair distance is made up from two radii)
        tm = Tol_matrix.from_radii([2 * r for r in RCORE.values()])
    stoichs = [
        ni
        for ni in product(stoichiometry, repeat=len(elements))
        if sum(ni) <= max_atoms
    ]
    if len(stoichs) == 0:
        warnings.warn(
            f"No valid stoichiometries for {elements}, {stoichiometry} <= {max_atoms}!"
        )
    store = []
    for num_ions in (bar := tqdm(stoichs)):
        if sum(num_ions) == 0:
            continue
        stoich = "".join(f"{s}{n}" for s, n in zip(elements, num_ions))
        bar.set_description(f"Stoichiometry {stoich}")

        def check_cell_shape(structure):
            # Want to avoid structures that are very long but narrow
            # vecs = np.linalg.norm(structure.cell.array, axis=-1)
            vecs = structure.cell.lengths()
            return vecs.max() / vecs.min() < 6

        # very few structures with low distances seem to slip through pyxtals checks, so double check here
        if min_dist is None:
            distance_filter = DistanceFilter()
        else:
            if isinstance(min_dist, float):
                distance_filter = DistanceFilter({e: min_dist / 2 for e in elements})
            elif isinstance(min_dist, dict):
                distance_filter = DistanceFilter(min_dist)
            else:
                assert (
                    False
                ), f"min_dist cannot by of type {type(min_dist)}: {min_dist}!"
        el, ni = zip(*((el, ni) for el, ni in zip(elements, num_ions) if ni > 0))
        # missing checker support
        # pr.create.structure.pyxtal(
        store += _pyxtal(
            group=range(1, 230 + 1),
            species=el,
            num_ions=ni,
            checker=lambda s: check_cell_shape(s) and distance_filter(s),
            factor=1.5,
            tm=tm,
        )
    return store