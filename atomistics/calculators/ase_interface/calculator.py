def get_potential_energy_from_ase(structure, ase_calculator_class, *args, **kwargs):
    structure.calc = ase_calculator_class(*args, **kwargs)
    return structure.get_potential_energy()


def get_forces_from_ase(structure, ase_calculator_class, *args, **kwargs):
    structure.calc = ase_calculator_class(*args, **kwargs)
    return structure.get_forces()


def evaluate_with_ase(task_dict, ase_calculator_class, *args, **kwargs):
    result_dict = {}
    if "calc_energy" in task_dict.keys():
        result_dict["energy"] = {
            k: get_potential_energy_from_ase(
                structure=v, ase_calculator_class=ase_calculator_class, *args, **kwargs
            )
            for k, v in task_dict["calc_energy"].items()
        }
    if "calc_forces" in task_dict.keys():
        result_dict["forces"] = {
            k: get_forces_from_ase(
                structure=v, ase_calculator_class=ase_calculator_class, *args, **kwargs
            )
            for k, v in task_dict["calc_forces"].items()
        }
    return result_dict
