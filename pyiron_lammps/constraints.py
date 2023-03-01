import numpy as np


def get_fixed_atom_boolean_vector(structure):
    fixed_atom_vector = np.array([[False, False, False]] * len(structure))
    for c in structure.constraints:
        c_dict = c.todict()
        if c_dict["name"] == "FixAtoms":
            fixed_atom_vector[c_dict["kwargs"]["indices"]] = [True, True, True]
        elif c_dict["name"] == "FixedPlane":
            if all(np.isin(c_dict["kwargs"]["direction"], [0, 1])):
                if "indices" in c_dict["kwargs"].keys():
                    fixed_atom_vector[c_dict["kwargs"]["indices"]] = np.array(
                        c_dict["kwargs"]["direction"]
                    ).astype(bool)
                elif "a" in c_dict["kwargs"].keys():
                    fixed_atom_vector[c_dict["kwargs"]["a"]] = np.array(
                        c_dict["kwargs"]["direction"]
                    ).astype(bool)
            else:
                raise ValueError(
                    "Currently the directions are limited to [1, 0, 0], [1, 1, 0], [1, 1, 1] and its permutations."
                )
        else:
            raise ValueError("Only FixAtoms and FixedPlane are currently supported. ")
    return fixed_atom_vector


def set_selective_dynamics(structure, calc_md):
    control_dict = {}
    if len(structure.constraints) > 0:
        sel_dyn = get_fixed_atom_boolean_vector(structure=structure)
        # Enter loop only if constraints present
        if len(np.argwhere(np.any(sel_dyn, axis=1)).flatten()) != 0:
            all_indices = np.arange(len(structure), dtype=int)
            constraint_xyz = np.argwhere(np.all(sel_dyn, axis=1)).flatten()
            not_constrained_xyz = np.setdiff1d(all_indices, constraint_xyz)
            # LAMMPS starts counting from 1
            constraint_xyz += 1
            ind_x = np.argwhere(sel_dyn[not_constrained_xyz, 0]).flatten()
            ind_y = np.argwhere(sel_dyn[not_constrained_xyz, 1]).flatten()
            ind_z = np.argwhere(sel_dyn[not_constrained_xyz, 2]).flatten()
            constraint_xy = not_constrained_xyz[np.intersect1d(ind_x, ind_y)] + 1
            constraint_yz = not_constrained_xyz[np.intersect1d(ind_y, ind_z)] + 1
            constraint_zx = not_constrained_xyz[np.intersect1d(ind_z, ind_x)] + 1
            constraint_x = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_x, ind_y), ind_z)] + 1
            )
            constraint_y = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_y, ind_z), ind_x)] + 1
            )
            constraint_z = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_z, ind_x), ind_y)] + 1
            )
            control_dict = {}
            if len(constraint_xyz) > 0:
                control_dict["group constraintxyz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_xyz]
                )
                control_dict["fix constraintxyz"] = "constraintxyz setforce 0.0 0.0 0.0"
                if calc_md:
                    control_dict["velocity constraintxyz"] = "set 0.0 0.0 0.0"
            if len(constraint_xy) > 0:
                control_dict["group constraintxy"] = "id " + " ".join(
                    [str(ind) for ind in constraint_xy]
                )
                control_dict["fix constraintxy"] = "constraintxy setforce 0.0 0.0 NULL"
                if calc_md:
                    control_dict["velocity constraintxy"] = "set 0.0 0.0 NULL"
            if len(constraint_yz) > 0:
                control_dict["group constraintyz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_yz]
                )
                control_dict["fix constraintyz"] = "constraintyz setforce NULL 0.0 0.0"
                if calc_md:
                    control_dict["velocity constraintyz"] = "set NULL 0.0 0.0"
            if len(constraint_zx) > 0:
                control_dict["group constraintxz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_zx]
                )
                control_dict["fix constraintxz"] = "constraintxz setforce 0.0 NULL 0.0"
                if calc_md:
                    control_dict["velocity constraintxz"] = "set 0.0 NULL 0.0"
            if len(constraint_x) > 0:
                control_dict["group constraintx"] = "id " + " ".join(
                    [str(ind) for ind in constraint_x]
                )
                control_dict["fix constraintx"] = "constraintx setforce 0.0 NULL NULL"
                if calc_md:
                    control_dict["velocity constraintx"] = "set 0.0 NULL NULL"
            if len(constraint_y) > 0:
                control_dict["group constrainty"] = "id " + " ".join(
                    [str(ind) for ind in constraint_y]
                )
                control_dict["fix constrainty"] = "constrainty setforce NULL 0.0 NULL"
                if calc_md:
                    control_dict["velocity constrainty"] = "set NULL 0.0 NULL"
            if len(constraint_z) > 0:
                control_dict["group constraintz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_z]
                )
                control_dict["fix constraintz"] = "constraintz setforce NULL NULL 0.0"
                if calc_md:
                    control_dict["velocity constraintz"] = "set NULL NULL 0.0"
    return control_dict
