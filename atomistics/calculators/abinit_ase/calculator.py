from ase.calculators.abinit import Abinit
from ase.units import Ry


def get_potential_energy_from_abinit(
    structure, label='abinit', nbands=32, ecut=10 * Ry, kpts=(5, 5, 5), toldfe=1.0e-2,
):
    structure.calc = Abinit(
        label=label,
        nbands=nbands,
        ecut=ecut,
        kpts=kpts,
        toldfe=toldfe,
    )
    return structure.get_potential_energy()


def get_forces_from_abinit(
    structure, label='abinit', nbands=32, ecut=10 * Ry, kpts=(5, 5, 5), toldfe=1.0e-2,
):
    structure.calc = Abinit(
        label=label,
        nbands=nbands,
        ecut=ecut,
        kpts=kpts,
        toldfe=toldfe,
    )
    return structure.get_forces()


def evaluate_with_abinit(
    task_dict, label='abinit', nbands=32, ecut=10 * Ry, kpts=(5, 5, 5), toldfe=1.0e-2,
):
    result_dict = {}
    if "calc_energy" in task_dict.keys():
        result_dict["energy"] = {
            k: get_potential_energy_from_abinit(
                structure=v,
                label=label,
                nbands=nbands,
                ecut=ecut,
                kpts=kpts,
                toldfe=toldfe,
            )
            for k, v in task_dict["calc_energy"].items()
        }
    if "calc_forces" in task_dict.keys():
        result_dict["forces"] = {
            k: get_forces_from_abinit(
                structure=v,
                label=label,
                nbands=nbands,
                ecut=ecut,
                kpts=kpts,
                toldfe=toldfe,
            )
            for k, v in task_dict["calc_forces"].items()
        }
    return result_dict
