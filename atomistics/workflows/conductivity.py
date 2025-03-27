from typing import Optional

from ase.atoms import Atoms
import numpy as np
from phono3py import Phono3py
from phonopy.units import VaspToTHz
import structuretoolkit

from atomistics.workflows.interface import Workflow


def generate_structures_helper(
    structure: Atoms,
    supercell_matrix: Optional[list] = None,
    primitive_matrix: Optional[list] = None,
    phonon_supercell_matrix: Optional[list] = None,
    mesh_numbers: Optional[list] = None,
    cutoff_frequency: float = 0.0001,
    frequency_factor_to_THz: float = VaspToTHz,
    is_symmetry: bool = True,
    is_mesh_symmetry: bool = True,
    use_grg: bool = False,
    SNF_coordinates: str = 'reciprocal',
    make_r0_average: bool = True,
    symprec: float = 1e-05,
    log_level: int = 0,
):
    if primitive_matrix is None:
        primitive_matrix = structuretoolkit.analyse.get_primitive_cell(structure).cell.array / np.diag(structure.cell.array).mean()
    if supercell_matrix is None:
        supercell_matrix = [2, 2, 2]
    if phonon_supercell_matrix is None:
        phonon_supercell_matrix = [4, 4, 4]
    if mesh_numbers is None:
        mesh_numbers = [11, 11, 11]
    phono = Phono3py(
        unitcell=structuretoolkit.common.atoms_to_phonopy(structure),
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
        cutoff_frequency=cutoff_frequency,
        frequency_factor_to_THz=frequency_factor_to_THz,
        is_symmetry=is_symmetry,
        is_mesh_symmetry=is_mesh_symmetry,
        use_grg=use_grg,
        SNF_coordinates=SNF_coordinates,
        make_r0_average=make_r0_average,
        symprec=symprec,
        calculator=None,
        log_level=log_level,
    )
    phono.mesh_numbers = mesh_numbers
    phono.generate_displacements()
    phono.generate_fc2_displacements()
    task_dict_lst = [
        {"calc_forces": structuretoolkit.common.phonopy_to_atoms(s)}
        for s in phono.supercells_with_displacements
    ]
    task_dict_phono_lst = [
        {"calc_forces": structuretoolkit.common.phonopy_to_atoms(s)}
        for s in phono.phonon_supercells_with_displacements
    ]
    supercell_count = len(task_dict_lst)
    phonocell_count = len(task_dict_phono_lst)
    return task_dict_lst + task_dict_phono_lst, supercell_count, phonocell_count, phono


def analyse_structures_helper(
    phono: Phono3py,
    forces_lst: list,
    supercell_count: int,
):
    phono.forces = np.array(forces_lst[:supercell_count])
    phono.phonon_forces = np.array(forces_lst[supercell_count:])
    phono.produce_fc2()
    phono.produce_fc3()
    phono.init_phph_interaction()
    phono.run_phonon_solver()
    phono.run_thermal_conductivity()
    return {
        "temperature": phono.thermal_conductivity.get_temperatures(),
        "kappa": phono.thermal_conductivity.kappa[0]
    }


class ConductivityWorkflow(Workflow):
    def __init__(
        self,
        structure: Atoms,
        supercell_matrix: Optional[list] = None,
        primitive_matrix: Optional[list] = None,
        phonon_supercell_matrix: Optional[list] = None,
        mesh_numbers: Optional[list] = None,
        cutoff_frequency: float = 0.0001,
        frequency_factor_to_THz: float = VaspToTHz,
        is_symmetry: bool = True,
        is_mesh_symmetry: bool = True,
        use_grg: bool = False,
        SNF_coordinates: str = 'reciprocal',
        make_r0_average: bool = True,
        symprec: float = 1e-05,
        log_level: int = 0,
    ):
        self._structure = structure
        self._supercell_matrix = supercell_matrix
        self._primitive_matrix = primitive_matrix
        self._phonon_supercell_matrix = phonon_supercell_matrix
        self._mesh_numbers = mesh_numbers
        self._cutoff_frequency = cutoff_frequency
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_symmetry = is_symmetry
        self._is_mesh_symmetry = is_mesh_symmetry
        self._use_grg = use_grg
        self._SNF_coordinates = SNF_coordinates
        self._make_r0_average = make_r0_average
        self._symprec = symprec
        self._log_level = log_level
        self._supercell_count = None
        self._phonocell_count = None
        self._phono = None

    def generate_structures(self) -> dict:
        task_dict_lst, supercell_count, phonocell_count, phono = generate_structures_helper(
            structure=self._structure,
            supercell_matrix=self._supercell_matrix,
            primitive_matrix=self._primitive_matrix,
            phonon_supercell_matrix=self._phonon_supercell_matrix,
            mesh_numbers=self._mesh_numbers,
            cutoff_frequency=self._cutoff_frequency,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            is_symmetry=self._is_symmetry,
            is_mesh_symmetry=self._is_mesh_symmetry,
            use_grg=self._use_grg,
            SNF_coordinates=self._SNF_coordinates,
            make_r0_average=self._make_r0_average,
            symprec=self._symprec,
            log_level=self._log_level,
        )
        self._supercell_count = supercell_count
        self._phonocell_count = phonocell_count
        self._phono = phono
        return task_dict_lst

    def analyse_structures(
        self, forces_lst: list
    ) -> dict:
        return analyse_structures_helper(
            phono=self._phono,
            forces_lst=forces_lst,
            supercell_count=self._supercell_count,
        )
