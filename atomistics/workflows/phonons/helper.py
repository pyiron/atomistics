from typing import Optional
import numpy as np
import scipy.constants


def get_supercell_matrix(interaction_range, cell):
    supercell_range = np.ceil(
        interaction_range / np.array([np.linalg.norm(vec) for vec in cell])
    )
    return np.eye(3) * supercell_range


def get_hesse_matrix(force_constants):
    """

    Returns:

    """
    unit_conversion = (
        scipy.constants.physical_constants["Hartree energy in eV"][0]
        / scipy.constants.physical_constants["Bohr radius"][0] ** 2
        * scipy.constants.angstrom**2
    )
    force_shape = np.shape(force_constants)
    force_reshape = force_shape[0] * force_shape[2]
    return (
        np.transpose(force_constants, (0, 2, 1, 3)).reshape(
            (force_reshape, force_reshape)
        )
        / unit_conversion
    )


def plot_dos(dos_energies, dos_total, *args, axis=None, **kwargs):
    """
    Plot the DOS.

    If "label" is present in `kwargs` a legend is added to the plot automatically.

    Args:
        axis (optional): matplotlib axis to use, if None create a new one
        *args: passed to `axis.plot`
        **kwargs: passed to `axis.plot`

    Returns:
        matplotlib.axes._subplots.AxesSubplot: axis with the plot
    """
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots(1, 1)
    axis.plot(dos_energies, dos_total, *args, **kwargs)
    axis.set_xlabel("Frequency [THz]")
    axis.set_ylabel("DOS")
    axis.set_title("Phonon DOS vs Energy")
    if "label" in kwargs:
        axis.legend()
    return axis


def get_band_structure(
    phonopy, npoints=101, with_eigenvectors=False, with_group_velocities=False
):
    """
    Calculate band structure with automatic path through reciprocal space.

    Can only be called after job is finished.

    Args:
        npoints (int, optional):  Number of sample points between high symmetry points.
        with_eigenvectors (boolean, optional):  Calculate eigenvectors, too
        with_group_velocities (boolean, optional):  Calculate group velocities, too

    Returns:
        :class:`dict` of the results from phonopy under the following keys
            - 'qpoints':  list of (npoints, 3), samples paths in reciprocal space
            - 'distances':  list of (npoints,), distance along the paths in reciprocal space
            - 'frequencies':  list of (npoints, band), phonon frequencies
            - 'eigenvectors':  list of (npoints, band, band//3, 3), phonon eigenvectors
            - 'group_velocities': list of (npoints, band), group velocities
        where band is the number of bands (number of atoms * 3).  Each entry is a list of arrays, and each array
        corresponds to one path between two high-symmetry points automatically picked by Phonopy and may be of
        different length than other paths.  As compared to the phonopy output this method also reshapes the
        eigenvectors so that they directly have the same shape as the underlying structure.

    Raises:
        :exception:`ValueError`: method is called on a job that is not finished or aborted
    """
    phonopy.auto_band_structure(
        npoints,
        with_eigenvectors=with_eigenvectors,
        with_group_velocities=with_group_velocities,
    )
    results = phonopy.get_band_structure_dict()
    if results["eigenvectors"] is not None:
        # see https://phonopy.github.io/phonopy/phonopy-module.html#eigenvectors for the way phonopy stores the
        # eigenvectors
        results["eigenvectors"] = [
            e.transpose(0, 2, 1).reshape(*e.shape[:2], -1, 3)
            for e in results["eigenvectors"]
        ]
    return results


def plot_band_structure(
    results,
    path_connections,
    labels,
    axis=None,
    *args,
    label: Optional[str] = None,
    **kwargs
):
    """
    Plot bandstructure calculated with :meth:`.get_bandstructure`.

    If :meth:`.get_bandstructure` hasn't been called before, it is automatically called with the default arguments.

    If `label` is passed a legend is added automatically.

    Args:
        axis (matplotlib axis, optional): plot to this axis, if not given a new one is created.
        *args: passed through to matplotlib.pyplot.plot when plotting the dispersion
        label (str, optional): label for dispersion line
        **kwargs: passed through to matplotlib.pyplot.plot when plotting the dispersion

    Returns:
        matplib axis: the axis the figure has been drawn to, if axis is given the same object is returned
    """
    # label is it's own argument because if you try to pass it via **kwargs every line would get the label giving a
    # messy legend
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots(1, 1)

    distances = results["distances"]
    frequencies = results["frequencies"]

    if "color" not in kwargs:
        kwargs["color"] = "black"

    offset = 0
    tick_positions = [distances[0][0]]
    for di, fi, ci in zip(distances, frequencies, path_connections):
        axis.axvline(tick_positions[-1], color="black", linestyle="dotted", alpha=0.5)
        line, *_ = axis.plot(offset + di, fi, *args, **kwargs)
        tick_positions.append(di[-1] + offset)
        if not ci:
            offset += 0.05
            axis.axvline(
                tick_positions[-1], color="black", linestyle="dotted", alpha=0.5
            )
            tick_positions.append(di[-1] + offset)
    if label is not None:
        line.set_label(label)
        axis.legend()
    axis.set_xticks(tick_positions[:-1])
    axis.set_xticklabels(labels)
    axis.set_xlabel("Bandpath")
    axis.set_ylabel("Frequency [THz]")
    axis.set_title("Bandstructure")
    return axis
