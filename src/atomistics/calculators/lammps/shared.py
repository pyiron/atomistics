from collections.abc import Iterable
from typing import Optional


def get_box_relax_command(
    pressure: float | Iterable[float | None], vmax: Optional[float]
) -> str:
    if not isinstance(pressure, Iterable):
        box_relax = f"fix ensemble all box/relax iso {pressure}"
    else:
        pressure_lst = list(pressure)
        if len(pressure_lst) == 3:
            pressure_str = " ".join(
                "{tag} {value}".format(tag=tag, value=value)
                for tag, value in zip(["x", "y", "z"], pressure_lst)
                if value is not None
            )
            box_relax = f"fix ensemble all box/relax {pressure_str}"
        elif len(pressure_lst) == 6:
            pressure_str = " ".join(
                "{tag} {value}".format(tag=tag, value=value)
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
            return box_relax + " vmax {vmax}".format(vmax=vmax)
        else:
            raise TypeError("vmax must be a float.")
    else:
        return box_relax
