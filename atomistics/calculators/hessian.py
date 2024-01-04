import numpy as np

from atomistics.calculators.wrapper import as_task_dict_evaluator


def check_force_constants(structure, force_constants):
    if structure is None:
        raise ValueError("Set reference structure via set_reference_structure() first")
    n_atom = len(structure.positions)
    if len(np.array([force_constants]).flatten()) == 1:
        return force_constants * np.eye(3 * n_atom)
    elif np.array(force_constants).shape == (3 * n_atom, 3 * n_atom):
        return force_constants
    elif np.array(force_constants).shape == (n_atom, n_atom):
        na = np.newaxis
        return (
            np.array(force_constants)[:, na, :, na] * np.eye(3)[na, :, na, :]
        ).flatten()
    elif len(np.shape(force_constants)) == 4:
        force_shape = np.shape(force_constants)
        if force_shape[2] == 3 and force_shape[3] == 3:
            force_reshape = force_shape[0] * force_shape[2]
            return np.transpose(force_constants, (0, 2, 1, 3)).reshape(
                (force_reshape, force_reshape)
            )
        elif force_shape[1] == 3 and force_shape[3] == 3:
            return np.array(force_constants).reshape(3 * n_atom, 3 * n_atom)
        else:
            raise AssertionError("force constant shape not recognized")
    else:
        raise AssertionError("force constant shape not recognized")


def get_displacement(structure_equilibrium, structure):
    displacements = structure.get_scaled_positions()
    displacements -= structure_equilibrium.get_scaled_positions()
    displacements -= np.rint(displacements)
    return np.einsum("ji,ni->nj", structure.cell, displacements)


def calc_forces_transformed(force_constants, structure_equilibrium, structure):
    displacements = get_displacement(structure_equilibrium, structure)
    position_transformed = displacements.reshape(
        displacements.shape[0] * displacements.shape[1]
    )
    return -np.dot(force_constants, position_transformed), displacements


def get_forces(force_constants, structure_equilibrium, structure):
    displacements = get_displacement(structure_equilibrium, structure)
    position_transformed = displacements.reshape(
        displacements.shape[0] * displacements.shape[1]
    )
    forces_transformed = -np.dot(force_constants, position_transformed)
    return forces_transformed.reshape(displacements.shape)


def get_energy_pot(
    force_constants, structure_equilibrium, structure, bulk_modulus=0, shear_modulus=0
):
    displacements = get_displacement(structure_equilibrium, structure)
    position_transformed = displacements.reshape(
        displacements.shape[0] * displacements.shape[1]
    )
    forces_transformed = -np.dot(force_constants, position_transformed)
    energy_pot = -1 / 2 * np.dot(position_transformed, forces_transformed)
    energy_pot += get_pressure_times_volume(
        stiffness_tensor=get_stiffness_tensor(
            bulk_modulus=bulk_modulus, shear_modulus=shear_modulus
        ),
        structure_equilibrium=structure_equilibrium,
        structure=structure,
    )
    return energy_pot


def get_stiffness_tensor(bulk_modulus=0, shear_modulus=0):
    stiffness_tensor = np.zeros((6, 6))
    stiffness_tensor[:3, :3] = bulk_modulus - 2 * shear_modulus / 3
    stiffness_tensor[:3, :3] += np.eye(3) * 2 * shear_modulus
    stiffness_tensor[3:, 3:] = np.eye(3) * shear_modulus
    return stiffness_tensor


def get_pressure_times_volume(stiffness_tensor, structure_equilibrium, structure):
    if np.sum(stiffness_tensor) != 0:
        epsilon = np.einsum(
            "ij,jk->ik",
            structure.cell,
            np.linalg.inv(structure_equilibrium.cell),
        ) - np.eye(3)
        epsilon = (epsilon + epsilon.T) * 0.5
        epsilon = np.append(epsilon.diagonal(), np.roll(epsilon, -1, axis=0).diagonal())
        pressure = -np.einsum("ij,j->i", stiffness_tensor, epsilon)
        pressure = pressure[3:] * np.roll(np.eye(3), -1, axis=1)
        pressure += pressure.T + np.eye(3) * pressure[:3]
        return -np.sum(epsilon * pressure) * structure.get_volume()
    else:
        return 0.0


@as_task_dict_evaluator
def evaluate_with_hessian(
    structure,
    tasks,
    structure_equilibrium,
    force_constants,
    bulk_modulus=0,
    shear_modulus=0,
):
    results = {}
    if "calc_energy" in tasks or "calc_forces" in tasks:
        force_constants = check_force_constants(
            structure=structure, force_constants=force_constants
        )
        if "calc_energy" in tasks:
            results["energy"] = get_energy_pot(
                structure=structure,
                structure_equilibrium=structure_equilibrium,
                force_constants=force_constants,
                bulk_modulus=bulk_modulus,
                shear_modulus=shear_modulus,
            )
        if "calc_forces" in tasks:
            results["forces"] = get_forces(
                structure=structure,
                structure_equilibrium=structure_equilibrium,
                force_constants=force_constants,
            )
    else:
        raise ValueError("The Hessian calculator does not implement:", tasks)
    return results
