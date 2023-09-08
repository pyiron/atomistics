from ase.calculators.emt import EMT


def get_potential_energy_from_emt(structure):
    structure.calc = EMT()
    return structure.get_potential_energy()


def get_forces_from_emt(structure):
    structure.calc = EMT()
    return structure.get_forces()


def evaluate_with_emt(task_dict):
    result_dict = {}
    if "calc_energy" in task_dict.keys():
        result_dict["energy"] = {
            k: get_potential_energy_from_emt(structure=v)
            for k, v in task_dict["calc_energy"].items()
        }
    if "calc_forces" in task_dict.keys():
        result_dict["forces"] = {
            k: get_forces_from_emt(structure=v)
            for k, v in task_dict["calc_forces"].items()
        }
    return result_dict
