"""
Defines (possibly) available calculator names, and gives access to evaluation functions
that use them (if they're available).

Intention is to give something easy to loop over in tests.

Warning: All tests using this infrastructure _must_ be run on pure aluminum structures.
"""
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from ase.build import bulk
from ase.calculators.abinit import Abinit
from ase.calculators.emt import EMT
from ase.calculators.espresso import Espresso
from ase.units import Ry

from atomistics.calculators.ase import evaluate_with_ase
from atomistics.calculators.interface import StrEnum

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from atomistics.calculators.interface import (
        ResultsDict,
        TaskDict,
        TaskDictEvaluator,
    )


class Calculators(StrEnum):
    abinit = "AbInit"
    emt = "EMT"
    gpaw = "GPaw"
    lammps = "Lammps"
    quantum_espresso = "Quantum Espresso"


def _evaluate_ase(ase_calculator: Calculator) -> TaskDictEvaluator:
    """An ASE task dict evaluator that already has the ASE calculator specified!"""
    def with_set_calculator(task_dict: TaskDict) -> ResultsDict:
        return evaluate_with_ase(task_dict, ase_calculator)
    return with_set_calculator


EVALUATION_FUNCTIONS: dict[Calculators: TaskDictEvaluator | None] = {}

# Ab Init
EVALUATION_FUNCTIONS[Calculators.abinit] = _evaluate_ase(
    Abinit(
        label='abinit_evcurve',
        nbands=32,
        ecut=10 * Ry,
        kpts=(3, 3, 3),
        toldfe=1.0e-2,
        v8_legacy_format=False,
    )
) if shutil.which("abinit") is not None else None

# EMT
EVALUATION_FUNCTIONS[Calculators.emt] = _evaluate_ase(EMT())

# GPAW
try:
    from gpaw import GPAW, PW
    EVALUATION_FUNCTIONS[Calculators.gpaw] = _evaluate_ase(
        GPAW(
            xc="PBE",
            mode=PW(300),
            kpts=(3, 3, 3)
        )
    )
except ImportError:
    EVALUATION_FUNCTIONS[Calculators.gpaw] = None

# Quantum Espresso
EVALUATION_FUNCTIONS[Calculators.quantum_espresso] = _evaluate_ase(
        Espresso(
            pseudopotentials={"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
            tstress=True,
            tprnfor=True,
            kpts=(3, 3, 3),
        )
    ) if shutil.which("pw.x") is not None else None

# Lammps
try:
    from atomistics.calculators.lammps import (
        evaluate_with_lammps, get_potential_dataframe
    )

    def evaluate_with_lammps_Al_potential(task_dict):
        potential = '1999--Mishin-Y--Al--LAMMPS--ipr1'
        resource_path = os.path.join(os.path.dirname(__file__), "static", "lammps")
        structure = bulk("Al", a=4.05, cubic=True)
        df_pot = get_potential_dataframe(
            structure=structure,
            resource_path=resource_path
        )
        df_pot_selected = df_pot[df_pot.Name == potential].iloc[0]

        return evaluate_with_lammps(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )

    EVALUATION_FUNCTIONS[Calculators.lammps] = evaluate_with_lammps_Al_potential
except ImportError:
    EVALUATION_FUNCTIONS[Calculators.lammps] = None
