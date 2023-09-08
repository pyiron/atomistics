from gpaw import GPAW, PW


def get_potential_energy_from_gpaw(structure, xc="PBE", encut=300, kpts=(5, 5, 5)):
    structure.calc = GPAW(xc=xc, mode=PW(encut), kpts=kpts)
    return structure.get_potential_energy()


def get_forces_from_gpaw(structure, xc="PBE", encut=300, kpts=(5, 5, 5)):
    structure.calc = structure.calc = GPAW(xc=xc, mode=PW(encut), kpts=kpts)
    return structure.get_forces()


def evaluate_with_gpaw(task_dict, xc="PBE", encut=300, kpts=(5, 5, 5)):
    result_dict = {}
    if "calc_energy" in task_dict.keys():
        result_dict["energy"] = {
            k: get_potential_energy_from_gpaw(
                structure=v, xc=xc, encut=encut, kpts=kpts
            )
            for k, v in task_dict["calc_energy"].items()
        }
    if "calc_forces" in task_dict.keys():
        result_dict["forces"] = {
            k: get_forces_from_gpaw(structure=v, xc=xc, encut=encut, kpts=kpts)
            for k, v in task_dict["calc_forces"].items()
        }
    return result_dict
