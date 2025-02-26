import os
from typing import Optional

import scipy.constants
from ase.atoms import Atoms
from sphinx_parser.ase import get_structure_group, id_spx_to_ase
from sphinx_parser.input import sphinx
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


def _write_input(
    structure: Atoms,
    working_directory: str,
    maxSteps: int = 100,
    eCut: float = 25.0,
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
        eCut=eCut,
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
    input_sx = sphinx.create(
        pawPot=pawPot_group,
        structure=struct_group,
        main=main_group,
        basis=basis_group,
        PAWHamiltonian=paw_group,
        initialGuess=initial_guess_group,
    )
    with open(os.path.join(working_directory, "input.sx"), "w") as f:
        f.write(to_sphinx(input_sx))


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


def calc_static_with_sphinxdft(
    structure: Atoms,
    working_directory: str,
    executable_function: callable,
    maxSteps: int = 100,
    eCut: float = 25,
    kpoint_coords: Optional[list[float, float, float]] = None,
    kpoint_folding: Optional[list[int, int, int]] = None,
    output_keys=OutputStatic.keys(),
) -> dict:
    if kpoint_coords is None:
        kpoint_coords = [0.5, 0.5, 0.5]
    if kpoint_folding is None:
        kpoint_folding = [3, 3, 3]
    _write_input(
        structure=structure,
        working_directory=working_directory,
        maxSteps=maxSteps,
        eCut=eCut,
        kpoint_coords=kpoint_coords,
        kpoint_folding=kpoint_folding,
    )
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
    maxSteps: int = 100,
    eCut: float = 25,
    kpoint_coords: Optional[list[float, float, float]] = None,
    kpoint_folding: Optional[list[int, int, int]] = None,
) -> dict:
    if kpoint_coords is None:
        kpoint_coords = [0.5, 0.5, 0.5]
    if kpoint_folding is None:
        kpoint_folding = [3, 3, 3]
    if "calc_energy" in tasks or "calc_forces" in tasks or "calc_stress" in tasks:
        return calc_static_with_sphinxdft(
            structure=structure,
            maxSteps=maxSteps,
            eCut=eCut,
            kpoint_coords=kpoint_coords,
            kpoint_folding=kpoint_folding,
            working_directory=working_directory,
            executable_function=executable_function,
            output_keys=get_quantities_from_tasks(tasks=tasks),
        )
    else:
        raise ValueError("The SphinxDFT calculator does not implement:", tasks)
