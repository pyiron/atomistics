static_calculation_output_keys = (
    "forces",  # np.ndarray (n, 3) [eV / Ang^2]
    "energy",  # float [eV]
    "stress",  # np.ndarray (3, 3) [GPa]
    "volume",  # float [Ang^3]
)


molecular_dynamics_output_keys = (
    "positions",  # np.ndarray (t, n, 3) [Ang]
    "cell",  # np.ndarray (t, 3, 3) [Ang]
    "forces",  # np.ndarray (t, n, 3) [eV / Ang^2]
    "temperature",  # np.ndarray (t) [K]
    "energy_pot",  # np.ndarray (t) [eV]
    "energy_tot",  # np.ndarray (t) [eV]
    "pressure",  # np.ndarray (t, 3, 3) [GPa]
    "velocities",  # np.ndarray (t, n, 3) [eV / Ang]
    "volume",  # np.ndarray (t) [Ang^3]
)


thermal_expansion_output_keys = (
    "temperatures",  # np.ndarray (T) [K]
    "volumes",  # np.ndarray (T) [Ang^3]
)


thermodynamic_output_keys = (
    "temperatures",  # np.ndarray (T) [K]
    "volumes",  # np.ndarray (T) [Ang^3]
    "free_energy",  # np.ndarray (T) [eV]
    "entropy",  # np.ndarray (T) [eV]
    "heat_capacity",  # np.ndarray (T) [eV]
)


energy_volume_curve_output_keys = (
    "energy_eq",  # float [eV]
    "volume_eq",  # float [Ang^3]
    "bulkmodul_eq",  # float [GPa]
    "b_prime_eq",  # float
    "fit_dict",  # dict
    "energy",  # np.ndarray (V) [eV]
    "volume",  # np.ndarray (V) [Ang^3]
)


elastic_matrix_output_keys = (
    "elastic_matrix",  # np.ndarray (6,6) [GPa]
    "elastic_matrix_inverse",  # np.ndarray (6,6) [GPa]
    "bulkmodul_voigt",  # float [GPa]
    "bulkmodul_reuss",  # float [GPa]
    "bulkmodul_hill",  # float [GPa]
    "shearmodul_voigt",  # float [GPa]
    "shearmodul_reuss",  # float [GPa]
    "shearmodul_hill",  # float [GPa]
    "youngsmodul_voigt",  # float [GPa]
    "youngsmodul_reuss",  # float [GPa]
    "youngsmodul_hill",  # float [GPa]
    "poissonsratio_voigt",  # float
    "poissonsratio_reuss",  # float
    "poissonsratio_hill",  # float
    "AVR",  # float
    "elastic_matrix_eigval",  # np.ndarray (6,6) [GPa]
)


phonon_output_keys = (
    "mesh_dict",  # dict
    "band_structure_dict",  # dict
    "total_dos_dict",  # dict
    "dynamical_matrix",  # dict
    "force_constants",  # dict
)
