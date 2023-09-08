from ase.calculators.espresso import Espresso


def get_potential_energy_from_qe(
    structure, pseudopotentials, tstress=True, tprnfor=True, kpts=(5, 5, 5)
):
    structure.calc = Espresso(
        pseudopotentials=pseudopotentials, tstress=tstress, tprnfor=tprnfor, kpts=kpts
    )
    return structure.get_potential_energy()


def get_forces_from_qe(
    structure, pseudopotentials, tstress=True, tprnfor=True, kpts=(5, 5, 5)
):
    structure.calc = Espresso(
        pseudopotentials=pseudopotentials, tstress=tstress, tprnfor=tprnfor, kpts=kpts
    )
    return structure.get_forces()


def evaluate_with_quantumespresso(
    task_dict, pseudopotentials, tstress=True, tprnfor=True, kpts=(5, 5, 5)
):
    result_dict = {}
    if "calc_energy" in task_dict.keys():
        result_dict["energy"] = {
            k: get_potential_energy_from_qe(
                structure=v,
                pseudopotentials=pseudopotentials,
                tstress=tstress,
                tprnfor=tprnfor,
                kpts=kpts,
            )
            for k, v in task_dict["calc_energy"].items()
        }
    if "calc_forces" in task_dict.keys():
        result_dict["forces"] = {
            k: get_forces_from_qe(
                structure=v,
                pseudopotentials=pseudopotentials,
                tstress=tstress,
                tprnfor=tprnfor,
                kpts=kpts,
            )
            for k, v in task_dict["calc_forces"].items()
        }
    return result_dict
