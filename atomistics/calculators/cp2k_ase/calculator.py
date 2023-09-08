from ase.calculators.cp2k import CP2K


def get_potential_energy_from_cp2k(
    structure,
    command="cp2k_shell.sopt",
    basis_set_file="./BASIS_SET",
    basis_set="DZVP-GTH-PADE",
    potential_file="./GTH_POTENTIALS",
    pseudo_potential="GTH-PADE-q4",
):
    structure.calc = CP2K(
        command=command,
        basis_set_file=basis_set_file,
        basis_set=basis_set,
        potential_file=potential_file,
        pseudo_potential=pseudo_potential,
    )
    return structure.get_potential_energy()


def get_forces_from_cp2k(
    structure,
    command="cp2k_shell.sopt",
    basis_set_file="./BASIS_SET",
    basis_set="DZVP-GTH-PADE",
    potential_file="./GTH_POTENTIALS",
    pseudo_potential="GTH-PADE-q4",
):
    structure.calc = CP2K(
        command=command,
        basis_set_file=basis_set_file,
        basis_set=basis_set,
        potential_file=potential_file,
        pseudo_potential=pseudo_potential,
    )
    return structure.get_forces()


def evaluate_with_cp2k(
    task_dict,
    command="cp2k_shell.sopt",
    basis_set_file="./BASIS_SET",
    basis_set="DZVP-GTH-PADE",
    potential_file="./GTH_POTENTIALS",
    pseudo_potential="GTH-PADE-q4",
):
    result_dict = {}
    if "calc_energy" in task_dict.keys():
        result_dict["energy"] = {
            k: get_potential_energy_from_cp2k(
                structure=v,
                command=command,
                basis_set_file=basis_set_file,
                basis_set=basis_set,
                potential_file=potential_file,
                pseudo_potential=pseudo_potential,
            )
            for k, v in task_dict["calc_energy"].items()
        }
    if "calc_forces" in task_dict.keys():
        result_dict["forces"] = {
            k: get_forces_from_cp2k(
                structure=v,
                command=command,
                basis_set_file=basis_set_file,
                basis_set=basis_set,
                potential_file=potential_file,
                pseudo_potential=pseudo_potential,
            )
            for k, v in task_dict["calc_forces"].items()
        }
    return result_dict
