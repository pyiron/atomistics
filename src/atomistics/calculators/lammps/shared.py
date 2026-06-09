from collections.abc import Iterable
from typing import Optional


def get_box_relax_command(
    pressure: float | Iterable[float | None], vmax: Optional[float]
) -> str:
    """
    Build a LAMMPS ``fix box/relax`` command string for the given pressure specification.

    When ``pressure`` is a scalar the command uses isotropic relaxation (``iso``).
    When ``pressure`` is an iterable of length 3, the x/y/z components are set
    individually; components that are ``None`` are omitted.  An iterable of length 6
    additionally sets the xy/xz/yz off-diagonal components.

    Args:
        pressure (float | Iterable[float | None]): Target pressure in bar.
            A scalar applies isotropic pressure; an iterable of 3 or 6 elements
            applies anisotropic pressure component-wise (``None`` entries are skipped).
        vmax (float | None): Maximum fractional volume change per timestep (LAMMPS
            ``vmax`` keyword).  No ``vmax`` clause is added when ``None``.

    Returns:
        str: A complete LAMMPS ``fix`` command string.

    Raises:
        ValueError: If ``pressure`` is an iterable whose length is not 3 or 6.
        TypeError: If ``vmax`` is not a ``float``.
    """
    if not isinstance(pressure, Iterable):
        box_relax = f"fix ensemble all box/relax iso {pressure}"
    else:
        pressure_lst = list(pressure)
        if len(pressure_lst) == 3:
            pressure_str = " ".join(
                f"{tag} {value}"
                for tag, value in zip(["x", "y", "z"], pressure_lst)
                if value is not None
            )
            box_relax = f"fix ensemble all box/relax {pressure_str}"
        elif len(pressure_lst) == 6:
            pressure_str = " ".join(
                f"{tag} {value}"
                for tag, value in zip(["x", "y", "z", "xy", "xz", "yz"], pressure_lst)
                if value is not None
            )
            box_relax = f"fix ensemble all box/relax {pressure_str}"
        else:
            raise ValueError(
                "pressure must be a float or an iterable of length 3 or 6."
            )
    if vmax is not None:
        if isinstance(vmax, float):
            return box_relax + f" vmax {vmax}"
        else:
            raise TypeError("vmax must be a float.")
    else:
        return box_relax
