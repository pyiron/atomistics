import os
from typing import Optional

import scipy.constants
from ase.atoms import Atoms
from sphinx_parser.ase import get_structure_group, id_spx_to_ase
from sphinx_parser.input import sphinx
from sphinx_parser.jobs import apply_minimization
from sphinx_parser.output import collect_energy_dat, collect_eval_forces
from sphinx_parser.potential import get_paw_from_structure
from sphinx_parser.toolkit import to_sphinx

from atomistics.calculators.interface import get_quantities_from_tasks
from atomistics.calculators.wrapper import as_task_dict_evaluator
from atomistics.shared.output import OutputStatic

BOHR_TO_ANGSTROM = (
    scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom
)
HARTREE_TO_EV = scipy.constants.physical_constants["Hartree energy in eV"][0]
HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


def _generate_input(
    structure: Atoms,
    maxSteps: int = 100,
    energy_cutoff_in_eV: float = 500.0,
    kpoint_coords: Optional[list[float, float, float]] = None,
    kpoint_folding: Optional[list[int, int, int]] = None,
):
    if kpoint_coords is None:
        kpoint_coords = [0.5, 0.5, 0.5]
    if kpoint_folding is None:
        kpoint_folding = [3, 3, 3]
    struct_group, spin_lst = get_structure_group(structure)
    main_group = sphinx.main.create(
        scfDiag=sphinx.main.scfDiag.create(maxSteps=maxSteps, blockCCG={}),
        evalForces=sphinx.main.evalForces.create(file="relaxHist.sx"),
    )
    pawPot_group = get_paw_from_structure(structure)
    basis_group = sphinx.basis.create(
        eCut=energy_cutoff_in_eV / HARTREE_TO_EV,
        kPoint=sphinx.basis.kPoint.create(coords=kpoint_coords),
        folding=kpoint_folding,
    )
    paw_group = sphinx.PAWHamiltonian.create(xc=1, spinPolarized=False, ekt=0.2)
    initial_guess_group = sphinx.initialGuess.create(
        waves=sphinx.initialGuess.waves.create(
            lcao=sphinx.initialGuess.waves.lcao.create()
        ),
        rho=sphinx.initialGuess.rho.create(atomicOrbitals=True),
    )
    return sphinx.create(
        pawPot=pawPot_group,
        structure=struct_group,
        main=main_group,
        basis=basis_group,
        PAWHamiltonian=paw_group,
        initialGuess=initial_guess_group,
    )


class OutputParser:
    def __init__(self, working_directory: str, structure: Atoms):
        self._working_directory = working_directory
        self._structure = structure

    def get_energy(self):
        return (
            collect_energy_dat(os.path.join(self._working_directory, "energy.dat"))[
                "scf_energy_int"
            ][-1][-1]
            * HARTREE_TO_EV
        )

    def get_forces(self):
        return (
            collect_eval_forces(os.path.join(self._working_directory, "relaxHist.sx"))[
                "forces"
            ][-1][id_spx_to_ase(self._structure)]
            * HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM
        )

    def get_volume(self):
        return self._structure.get_volume()

    def get_stress(self):
        raise NotImplementedError()


def optimize_positions_with_sphinxdft(
    structure: Atoms,
    working_directory: str,
    executable_function: callable,
    max_electronic_steps: int = 100,
    energy_cutoff_in_eV: float = 500.0,
    mode: str = "linQN",
    dEnergy: float = 1.0e-6,
    max_ionic_steps: int = 50,
    kpoint_coords: Optional[list[float, float, float]] = None,
    kpoint_folding: Optional[list[int, int, int]] = None,
) -> Atoms:
    if kpoint_coords is None:
        kpoint_coords = [0.5, 0.5, 0.5]
    if kpoint_folding is None:
        kpoint_folding = [3, 3, 3]
    input_sx = _generate_input(
        structure=structure,
        maxSteps=max_electronic_steps,
        energy_cutoff_in_eV=energy_cutoff_in_eV,
        kpoint_coords=kpoint_coords,
        kpoint_folding=kpoint_folding,
    )
    input_sx = apply_minimization(
        sphinx_input=input_sx,
        mode=mode,
        dEnergy=dEnergy,
        maxSteps=max_ionic_steps,
    )
    with open(os.path.join(working_directory, "input.sx"), "w") as f:
        f.write(to_sphinx(input_sx))
    executable_function(working_directory)
    structure_copy = structure.copy()
    structure_copy.positions = (
        collect_eval_forces(os.path.join(working_directory, "relaxHist.sx"))[
            "positions"
        ][-1][id_spx_to_ase(structure)]
        / BOHR_TO_ANGSTROM
    )
    return structure_copy


def calc_static_with_sphinxdft(
    structure: Atoms,
    working_directory: str,
    executable_function: callable,
    max_electronic_steps: int = 100,
    energy_cutoff_in_eV: float = 500.0,
    kpoint_coords: Optional[list[float, float, float]] = None,
    kpoint_folding: Optional[list[int, int, int]] = None,
    output_keys=OutputStatic.keys(),
) -> dict:
    if kpoint_coords is None:
        kpoint_coords = [0.5, 0.5, 0.5]
    if kpoint_folding is None:
        kpoint_folding = [3, 3, 3]
    input_sx = _generate_input(
        structure=structure,
        maxSteps=max_electronic_steps,
        energy_cutoff_in_eV=energy_cutoff_in_eV,
        kpoint_coords=kpoint_coords,
        kpoint_folding=kpoint_folding,
    )
    with open(os.path.join(working_directory, "input.sx"), "w") as f:
        f.write(to_sphinx(input_sx))
    executable_function(working_directory)
    output_obj = OutputParser(working_directory=working_directory, structure=structure)
    result_dict = OutputStatic(
        forces=output_obj.get_forces,
        energy=output_obj.get_energy,
        stress=output_obj.get_stress,
        volume=output_obj.get_volume,
    ).get(output_keys=output_keys)
    return result_dict


@as_task_dict_evaluator
def evaluate_with_sphinx(
    structure: Atoms,
    tasks: list,
    working_directory: str,
    executable_function: callable,
    max_electronic_steps: int = 100,
    energy_cutoff_in_eV: float = 500,
    kpoint_coords: Optional[list[float, float, float]] = None,
    kpoint_folding: Optional[list[int, int, int]] = None,
    sphinx_optimizer_kwargs: Optional[dict] = None,
) -> dict:
    if kpoint_coords is None:
        kpoint_coords = [0.5, 0.5, 0.5]
    if kpoint_folding is None:
        kpoint_folding = [3, 3, 3]
    if sphinx_optimizer_kwargs is None:
        sphinx_optimizer_kwargs = {}
    results = {}
    if "optimize_positions" in tasks:
        results["structure_with_optimized_positions"] = (
            optimize_positions_with_sphinxdft(
                structure=structure,
                max_electronic_steps=max_electronic_steps,
                energy_cutoff_in_eV=energy_cutoff_in_eV,
                kpoint_coords=kpoint_coords,
                kpoint_folding=kpoint_folding,
                working_directory=working_directory,
                executable_function=executable_function,
                **sphinx_optimizer_kwargs,
            )
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_sphinxdft(
            structure=structure,
            max_electronic_steps=max_electronic_steps,
            energy_cutoff_in_eV=energy_cutoff_in_eV,
            kpoint_coords=kpoint_coords,
            kpoint_folding=kpoint_folding,
            working_directory=working_directory,
            executable_function=executable_function,
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The SphinxDFT calculator does not implement:", tasks)
    return results
