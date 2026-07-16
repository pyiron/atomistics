# LAMMPS Interactive-Session Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three low-level LAMMPS interactive-session functions to `atomistics.calculators.lammps` (instantaneous potential energy, time-averaged potential energy via a LAMMPS fix, and a full cell/positions/velocities structure snapshot) so that hybrid MD/Monte-Carlo workflows can be built on the public `atomistics` API instead of raw `pylammpsmpi` calls, then port `notebooks/aucu_dist_v9.ipynb` onto them.

**Architecture:** Three new functions are added to `atomistics/calculators/lammps/libcalculator.py`, following the existing `_with_lammpslib` naming convention and built on the same `LammpsASELibrary` session object the rest of the module already uses. One new Jinja2 template constant (`LAMMPS_AVE_ENERGY`) is added to `commands.py`, following the existing template-constant pattern. The already-implemented `lammps_run`/`lammps_shutdown` helpers (currently internal to `helpers.py`) are exported publicly alongside the three new functions. No new abstractions are introduced beyond this — the MC-swap loop logic stays in the notebook.

**Tech Stack:** Python, `pylammpsmpi.LammpsASELibrary`, `jinja2.Template`, `ase.Atoms`, `pytest`/`unittest`, `nbformat`/`jupyter nbconvert` for the notebook.

## Global Constraints

- Follow the existing `_with_lammpslib` naming suffix for all new public functions in `libcalculator.py`.
- All new tests go in `tests/`, use `unittest.TestCase`, and are gated with the existing `@unittest.skipIf(skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped.")` pattern (see `tests/test_lammpslib_md.py`).
- Use the potential `"1999--Mishin-Y--Al--LAMMPS--ipr1"` with `resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps")` for all new tests (matches every existing LAMMPS test in `tests/`), retrieved via `get_potential_by_name`.
- New functions must not require any dependency beyond what `atomistics[lammps]` already installs (`pylammpsmpi`, `jinja2`, `lammpsparser`).
- Run the full test command `python -m pytest tests/test_lammps_interactive_session.py -v` after every task; do not proceed to the next task if it fails, except for the pre-existing environment issue described below.
- This work happens in the git worktree at `/Users/janssen/projects/atomistics/.worktrees/lammps-interactive-session` (branch `lammps-interactive-session`) — always `cd` there first, never touch the primary checkout at `/Users/janssen/projects/atomistics`. The editable install must point at this worktree's `src/` — verify with `python3 -c "import atomistics; print(atomistics.__file__)"`; it must print a path under `/Users/janssen/projects/atomistics/.worktrees/lammps-interactive-session/src/atomistics/__init__.py`. If it doesn't, run `python3 -m pip install -e . --no-deps -q` from the worktree root first.
- **Known pre-existing environment issue (not caused by this feature, do not attempt to fix it):** the installed `lammps` conda package (2024.08.29) predates a `pylammpsmpi==0.5.0` API change, so *every* LAMMPS session created via `interactive_structure_setter` — old code and any new code from this plan alike — currently fails with `TypeError: lammps.create_atoms() got an unexpected keyword argument 'atomid'`. This already breaks ~30 tests on `main` before this plan's changes. Any test in this plan that builds a live session (via `lammps_run`, directly or through `calc_molecular_dynamics_npt_with_lammpslib`, etc.) will hit this same `TypeError`. When that happens: confirm the failure message is *exactly* this `TypeError` (proving your new code was reached correctly and the failure is the known environment issue, not a bug in your work); do not treat it as a task failure; report it in your DONE/DONE_WITH_CONCERNS summary with the exact error text as evidence. A test failing for any *other* reason (ImportError, AttributeError, wrong value, different TypeError) is a real bug in your task and must be fixed before reporting done. Tests that only import symbols or call code paths that never construct a live session (e.g. Task 4's export test) must still fully PASS.

---

## Task 1: `get_energy_pot_with_lammpslib`

**Files:**
- Modify: `src/atomistics/calculators/lammps/libcalculator.py` (insert after line 690, the end of `calc_molecular_dynamics_thermal_expansion_with_lammpslib`, before line 693's `@as_task_dict_evaluator`)
- Create: `tests/test_lammps_interactive_session.py`

**Interfaces:**
- Produces: `get_energy_pot_with_lammpslib(lmp: LammpsASELibrary, run: int = 0) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lammps_interactive_session.py`:

```python
import os
import unittest

import numpy as np
from ase.build import bulk
from jinja2 import Template

try:
    from atomistics.calculators.lammps import get_potential_by_name, lammps_run, lammps_shutdown
    from atomistics.calculators.lammps.libcalculator import get_energy_pot_with_lammpslib
    from atomistics.calculators.lammps.commands import (
        LAMMPS_ENSEMBLE_NVT,
        LAMMPS_THERMO,
        LAMMPS_THERMO_STYLE,
        LAMMPS_TIMESTEP,
        LAMMPS_VELOCITY,
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


def _nvt_template(thermo=10, timestep=0.001, T=300.0, Tdamp=0.1, seed=4928459, include_velocity=True):
    parts = [LAMMPS_THERMO_STYLE, LAMMPS_TIMESTEP, LAMMPS_THERMO]
    if include_velocity:
        parts.append(LAMMPS_VELOCITY)
    parts.append(LAMMPS_ENSEMBLE_NVT)
    return Template("\n".join(parts)).render(
        thermo=thermo,
        timestep=timestep,
        temp=T,
        Tstart=T,
        Tstop=T,
        Tdamp=Tdamp,
        seed=seed,
        dist="gaussian",
        velocity_rescale_factor=2.0,
    )


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLammpsInteractiveSession(unittest.TestCase):
    def _get_potential(self):
        return get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )

    def test_get_energy_pot_matches_direct_getter(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = self._get_potential()
        lmp = lammps_run(
            structure=structure,
            potential_dataframe=df_pot_selected,
            input_template=_nvt_template(),
        )
        energy = get_energy_pot_with_lammpslib(lmp=lmp, run=10)
        self.assertIsInstance(energy, float)
        self.assertAlmostEqual(energy, lmp.interactive_energy_pot_getter(), places=8)
        lammps_shutdown(lmp_instance=lmp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lammps_interactive_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_energy_pot_with_lammpslib' from 'atomistics.calculators.lammps.libcalculator'`

- [ ] **Step 3: Implement `get_energy_pot_with_lammpslib`**

In `src/atomistics/calculators/lammps/libcalculator.py`, insert immediately after the end of `calc_molecular_dynamics_thermal_expansion_with_lammpslib` (after the closing of that function's `return`, before the `@as_task_dict_evaluator` line):

```python
def get_energy_pot_with_lammpslib(lmp: LammpsASELibrary, run: int = 0) -> float:
    """
    Advance an existing LAMMPS session by ``run`` timesteps and read back the
    instantaneous potential energy.

    Intended for hybrid MD/Monte-Carlo workflows that hold a session open across
    many small steps (see ``lammps_run`` / ``lammps_shutdown`` for building and
    tearing down such a session).

    Args:
        lmp (LammpsASELibrary): An active LAMMPS library instance.
        run (int): Number of timesteps to advance before reading the energy. Defaults to ``0``.

    Returns:
        float: The instantaneous potential energy in eV.
    """
    lmp.interactive_lib_command(f"run {run}")
    return lmp.interactive_energy_pot_getter()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lammps_interactive_session.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/atomistics/calculators/lammps/libcalculator.py tests/test_lammps_interactive_session.py
git commit -m "Add get_energy_pot_with_lammpslib for interactive LAMMPS sessions"
```

---

## Task 2: `LAMMPS_AVE_ENERGY` template + `get_energy_pot_average_with_lammpslib`

**Files:**
- Modify: `src/atomistics/calculators/lammps/commands.py` (append at end of file)
- Modify: `src/atomistics/calculators/lammps/libcalculator.py` (insert after `get_energy_pot_with_lammpslib`)
- Modify: `tests/test_lammps_interactive_session.py` (add test)

**Interfaces:**
- Consumes: `get_energy_pot_with_lammpslib` from Task 1 (not called directly, but same file/session pattern)
- Produces: `LAMMPS_AVE_ENERGY` (str, Jinja2 template with `{{energy_variable}}`, `{{fix_id}}`, `{{window}}`), `get_energy_pot_average_with_lammpslib(lmp: LammpsASELibrary, run: int, fix_id: str = "avePE") -> float`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_lammps_interactive_session.py`, add this import at the top alongside the existing ones from `atomistics.calculators.lammps.libcalculator`:

```python
    from atomistics.calculators.lammps.libcalculator import (
        get_energy_pot_average_with_lammpslib,
        get_energy_pot_with_lammpslib,
    )
    from atomistics.calculators.lammps.commands import (
        LAMMPS_AVE_ENERGY,
        LAMMPS_ENSEMBLE_NVT,
        LAMMPS_THERMO,
        LAMMPS_THERMO_STYLE,
        LAMMPS_TIMESTEP,
        LAMMPS_VELOCITY,
    )
```

(This replaces the two separate `libcalculator`/`commands` import blocks from Task 1 with these combined ones — same module paths, just importing more names.)

Add a template-builder helper and the test itself, appended after the `_nvt_template` function:

```python
def _nvt_with_average_template(window, T=300.0, timestep=0.001, Tdamp=0.1, seed=4928459, include_velocity=True, fix_id="avePE"):
    base = _nvt_template(
        thermo=window, timestep=timestep, T=T, Tdamp=Tdamp, seed=seed, include_velocity=include_velocity
    )
    ave_energy = Template(LAMMPS_AVE_ENERGY).render(
        energy_variable="myPE", fix_id=fix_id, window=window
    )
    return base + "\n" + ave_energy
```

Add inside `TestLammpsInteractiveSession`:

```python
    def test_get_energy_pot_average_is_close_to_instantaneous(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = self._get_potential()
        window = 10
        lmp = lammps_run(
            structure=structure,
            potential_dataframe=df_pot_selected,
            input_template=_nvt_with_average_template(window=window),
        )
        average_energy = get_energy_pot_average_with_lammpslib(lmp=lmp, run=window)
        instantaneous_energy = lmp.interactive_energy_pot_getter()
        self.assertIsInstance(average_energy, float)
        self.assertTrue(np.isfinite(average_energy))
        self.assertNotEqual(average_energy, 0.0)
        self.assertAlmostEqual(average_energy, instantaneous_energy, delta=5.0)
        lammps_shutdown(lmp_instance=lmp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lammps_interactive_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'LAMMPS_AVE_ENERGY' from 'atomistics.calculators.lammps.commands'`

- [ ] **Step 3: Implement the template and the function**

Append to `src/atomistics/calculators/lammps/commands.py`:

```python


LAMMPS_AVE_ENERGY = """\
variable {{energy_variable}} equal pe
fix {{fix_id}} all ave/time 1 {{window}} {{window}} v_{{energy_variable}}"""
```

In `src/atomistics/calculators/lammps/libcalculator.py`, insert immediately after `get_energy_pot_with_lammpslib`:

```python
def get_energy_pot_average_with_lammpslib(
    lmp: LammpsASELibrary, run: int, fix_id: str = "avePE"
) -> float:
    """
    Advance an existing LAMMPS session by ``run`` timesteps and read back the
    time-averaged potential energy from a LAMMPS ``fix ave/time`` (see
    ``LAMMPS_AVE_ENERGY`` for the matching input template).

    This reaches into the private ``lmp._interactive_library`` attribute to call
    ``extract_fix`` directly, since ``pylammpsmpi`` does not expose LAMMPS'
    ``extract_fix`` functionality through its public API. It therefore only works
    when ``lmp`` was created with ``cores=1`` and without an ``executor`` argument
    -- the case in which ``pylammpsmpi`` backs the session with the raw
    ``lammps.lammps`` object rather than an MPI/executor wrapper. ``run`` should
    equal (or be an integer multiple of) the averaging window the ``fix ave/time``
    was configured with, otherwise the returned value reflects a stale or
    incomplete window.

    Args:
        lmp (LammpsASELibrary): An active LAMMPS library instance created with
            ``cores=1`` and no ``executor``, with a ``fix ave/time`` of id
            ``fix_id`` already defined (see ``LAMMPS_AVE_ENERGY``).
        run (int): Number of timesteps to advance before reading the averaged energy.
        fix_id (str): LAMMPS id of the ``fix ave/time`` to read from. Defaults to ``"avePE"``.

    Returns:
        float: The time-averaged potential energy in eV.
    """
    lmp.interactive_lib_command(f"run {run}")
    return lmp._interactive_library.extract_fix(fix_id, 0, 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lammps_interactive_session.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atomistics/calculators/lammps/commands.py src/atomistics/calculators/lammps/libcalculator.py tests/test_lammps_interactive_session.py
git commit -m "Add LAMMPS_AVE_ENERGY template and get_energy_pot_average_with_lammpslib"
```

---

## Task 3: `get_structure_snapshot_with_lammpslib`

**Files:**
- Modify: `src/atomistics/calculators/lammps/libcalculator.py` (insert after `get_energy_pot_average_with_lammpslib`)
- Modify: `tests/test_lammps_interactive_session.py` (add test)

**Interfaces:**
- Consumes: `lammps_run`, `_nvt_template` (test helper from Task 1), `get_energy_pot_with_lammpslib` (Task 1)
- Produces: `get_structure_snapshot_with_lammpslib(structure: Atoms, lmp: LammpsASELibrary) -> Atoms`

- [ ] **Step 1: Write the failing test**

Add `get_structure_snapshot_with_lammpslib` to the `libcalculator` import block in `tests/test_lammps_interactive_session.py`:

```python
    from atomistics.calculators.lammps.libcalculator import (
        get_energy_pot_average_with_lammpslib,
        get_energy_pot_with_lammpslib,
        get_structure_snapshot_with_lammpslib,
    )
```

Add inside `TestLammpsInteractiveSession`:

```python
    def test_snapshot_round_trip(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        df_pot_selected = self._get_potential()
        lmp = lammps_run(
            structure=structure,
            potential_dataframe=df_pot_selected,
            input_template=_nvt_template(),
        )
        get_energy_pot_with_lammpslib(lmp=lmp, run=10)
        snapshot = get_structure_snapshot_with_lammpslib(structure=structure, lmp=lmp)

        self.assertEqual(
            list(snapshot.get_chemical_symbols()), list(structure.get_chemical_symbols())
        )
        np.testing.assert_allclose(snapshot.cell.array, lmp.interactive_cells_getter())
        np.testing.assert_allclose(snapshot.positions, lmp.interactive_positions_getter())
        np.testing.assert_allclose(
            snapshot.get_velocities(), lmp.interactive_velocities_getter()
        )
        energy_before_restore = lmp.interactive_energy_pot_getter()
        lammps_shutdown(lmp_instance=lmp)

        # Resume from the snapshot without re-randomising velocities: a fresh
        # `velocity create` would overwrite the velocities the snapshot carries.
        resume_template = _nvt_template(include_velocity=False)
        lmp_resumed = lammps_run(
            structure=snapshot,
            potential_dataframe=df_pot_selected,
            input_template=resume_template,
        )
        energy_after_restore = get_energy_pot_with_lammpslib(lmp=lmp_resumed, run=0)
        self.assertAlmostEqual(energy_before_restore, energy_after_restore, places=6)
        lammps_shutdown(lmp_instance=lmp_resumed)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lammps_interactive_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_structure_snapshot_with_lammpslib' from 'atomistics.calculators.lammps.libcalculator'`

- [ ] **Step 3: Implement the function**

In `src/atomistics/calculators/lammps/libcalculator.py`, insert immediately after `get_energy_pot_average_with_lammpslib`:

```python
def get_structure_snapshot_with_lammpslib(structure: Atoms, lmp: LammpsASELibrary) -> Atoms:
    """
    Capture the current cell, positions, and velocities of a live LAMMPS session as a
    standalone ``ase.Atoms`` snapshot.

    The returned snapshot can be passed back into ``lammps_run`` (with an input
    template that does not re-initialise velocities, e.g. omitting the
    ``LAMMPS_VELOCITY`` command) to resume the session state later -- there is no
    ``pylammpsmpi`` API to set velocities on an existing session directly, so a full
    session rebuild from such a snapshot is the only way to restore them.

    Chemical symbols are copied unchanged from ``structure``; callers that need to
    change species (e.g. after a Monte Carlo swap) should call
    ``structure.set_chemical_symbols(...)`` before passing ``structure`` in.

    Args:
        structure (Atoms): Template structure to copy (species, constraints, etc.);
            its cell, positions, and velocities are overwritten from ``lmp``.
        lmp (LammpsASELibrary): An active LAMMPS library instance.

    Returns:
        Atoms: A new ``Atoms`` snapshot of the current session state.
    """
    snapshot = structure.copy()
    snapshot.set_cell(lmp.interactive_cells_getter(), scale_atoms=False)
    snapshot.set_positions(lmp.interactive_positions_getter())
    snapshot.set_velocities(lmp.interactive_velocities_getter())
    return snapshot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lammps_interactive_session.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atomistics/calculators/lammps/libcalculator.py tests/test_lammps_interactive_session.py
git commit -m "Add get_structure_snapshot_with_lammpslib for interactive LAMMPS sessions"
```

---

## Task 4: Export the new functions and `lammps_run`/`lammps_shutdown` publicly

**Files:**
- Modify: `src/atomistics/calculators/lammps/__init__.py`
- Modify: `src/atomistics/calculators/__init__.py`
- Modify: `tests/test_lammps_interactive_session.py` (switch imports to the public path, add signature test)

**Interfaces:**
- Consumes: all four functions from Tasks 1-3 plus existing `lammps_run`/`lammps_shutdown` (`src/atomistics/calculators/lammps/helpers.py`)
- Produces: `get_energy_pot_with_lammpslib`, `get_energy_pot_average_with_lammpslib`, `get_structure_snapshot_with_lammpslib`, `lammps_run`, `lammps_shutdown` importable from both `atomistics.calculators.lammps` and `atomistics.calculators`

- [ ] **Step 1: Write the failing test**

Replace the two import blocks in `tests/test_lammps_interactive_session.py` (the `from atomistics.calculators.lammps import ...` line and the `from atomistics.calculators.lammps.libcalculator import ...` line) with imports from the top-level public path only:

```python
try:
    from atomistics.calculators import (
        get_energy_pot_average_with_lammpslib,
        get_energy_pot_with_lammpslib,
        get_potential_by_name,
        get_structure_snapshot_with_lammpslib,
        lammps_run,
        lammps_shutdown,
    )
    from atomistics.calculators.lammps.commands import (
        LAMMPS_AVE_ENERGY,
        LAMMPS_ENSEMBLE_NVT,
        LAMMPS_THERMO,
        LAMMPS_THERMO_STYLE,
        LAMMPS_TIMESTEP,
        LAMMPS_VELOCITY,
    )

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True
```

(This deliberately imports from `atomistics.calculators`, the top-level package, rather than the `atomistics.calculators.lammps` submodule used in Tasks 1-3, to prove the new functions reached the public API described in the design doc.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lammps_interactive_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_energy_pot_with_lammpslib' from 'atomistics.calculators'`

- [ ] **Step 3: Export the functions**

In `src/atomistics/calculators/lammps/__init__.py`, replace the whole file with:

```python
from lammpsparser import get_potential_by_name, get_potential_dataframe

from atomistics.calculators.lammps.helpers import lammps_run, lammps_shutdown
from atomistics.calculators.lammps.libcalculator import (
    calc_molecular_dynamics_langevin_with_lammpslib,
    calc_molecular_dynamics_nph_with_lammpslib,
    calc_molecular_dynamics_npt_with_lammpslib,
    calc_molecular_dynamics_nvt_with_lammpslib,
    calc_molecular_dynamics_thermal_expansion_with_lammpslib,
    calc_static_with_lammpslib,
    evaluate_with_lammpslib,
    evaluate_with_lammpslib_library_interface,
    get_energy_pot_average_with_lammpslib,
    get_energy_pot_with_lammpslib,
    get_structure_snapshot_with_lammpslib,
    optimize_positions_and_volume_with_lammpslib,
    optimize_positions_with_lammpslib,
)
from atomistics.shared.import_warning import raise_warning

__all__: list[str] = [
    "calc_molecular_dynamics_thermal_expansion_with_lammpslib",
    "calc_molecular_dynamics_nph_with_lammpslib",
    "calc_molecular_dynamics_npt_with_lammpslib",
    "calc_molecular_dynamics_nvt_with_lammpslib",
    "calc_molecular_dynamics_langevin_with_lammpslib",
    "calc_static_with_lammpslib",
    "evaluate_with_lammpslib",
    "evaluate_with_lammpslib_library_interface",
    "get_energy_pot_average_with_lammpslib",
    "get_energy_pot_with_lammpslib",
    "get_structure_snapshot_with_lammpslib",
    "lammps_run",
    "lammps_shutdown",
    "optimize_positions_and_volume_with_lammpslib",
    "optimize_positions_with_lammpslib",
    "get_potential_dataframe",
    "get_potential_by_name",
]
lammps_phonon_functions: list[str] = ["calc_molecular_dynamics_phonons_with_lammpslib"]


try:
    from atomistics.calculators.lammps.phonon import (
        calc_molecular_dynamics_phonons_with_lammpslib,
    )
except ImportError as e:
    raise_warning(module_list=lammps_phonon_functions, import_error=e)
else:
    __all__ += lammps_phonon_functions
```

In `src/atomistics/calculators/__init__.py`:

1. Replace the `lammps_functions` list (currently lines 25-38) with:

```python
lammps_functions: list[str] = [
    "calc_molecular_dynamics_thermal_expansion_with_lammpslib",
    "calc_molecular_dynamics_nph_with_lammpslib",
    "calc_molecular_dynamics_npt_with_lammpslib",
    "calc_molecular_dynamics_nvt_with_lammpslib",
    "calc_molecular_dynamics_langevin_with_lammpslib",
    "calc_static_with_lammpslib",
    "evaluate_with_lammpslib",
    "evaluate_with_lammpslib_library_interface",
    "get_energy_pot_average_with_lammpslib",
    "get_energy_pot_with_lammpslib",
    "get_potential_dataframe",
    "get_potential_by_name",
    "get_structure_snapshot_with_lammpslib",
    "lammps_run",
    "lammps_shutdown",
    "optimize_positions_and_volume_with_lammpslib",
    "optimize_positions_with_lammpslib",
]
```

2. Replace the `from atomistics.calculators.lammps import (...)` block (currently lines 69-82) with:

```python
    from atomistics.calculators.lammps import (
        calc_molecular_dynamics_langevin_with_lammpslib,
        calc_molecular_dynamics_nph_with_lammpslib,
        calc_molecular_dynamics_npt_with_lammpslib,
        calc_molecular_dynamics_nvt_with_lammpslib,
        calc_molecular_dynamics_thermal_expansion_with_lammpslib,
        calc_static_with_lammpslib,
        evaluate_with_lammpslib,
        evaluate_with_lammpslib_library_interface,
        get_energy_pot_average_with_lammpslib,
        get_energy_pot_with_lammpslib,
        get_potential_by_name,
        get_potential_dataframe,
        get_structure_snapshot_with_lammpslib,
        lammps_run,
        lammps_shutdown,
        optimize_positions_and_volume_with_lammpslib,
        optimize_positions_with_lammpslib,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lammps_interactive_session.py -v`
Expected: PASS (3 tests)

Then run the full LAMMPS test suite to make sure nothing else broke:

Run: `python -m pytest tests/ -k lammps -v`
Expected: all tests PASS (no failures, no new errors)

- [ ] **Step 5: Commit**

```bash
git add src/atomistics/calculators/lammps/__init__.py src/atomistics/calculators/__init__.py tests/test_lammps_interactive_session.py
git commit -m "Export LAMMPS interactive-session primitives from the public API"
```

---

## Task 5: Port `notebooks/aucu_dist_v9.ipynb` to the new functions

**Files:**
- Create: `notebooks/aucu_dist_v9_atomistics.ipynb`
- Create (temporary, delete after use): `/tmp/build_aucu_notebook.py` (or use the scratchpad directory) — a script that builds the notebook via `nbformat` so cell source is easy to review and edit; not part of the deliverable.

**Interfaces:**
- Consumes: `get_energy_pot_with_lammpslib`, `get_energy_pot_average_with_lammpslib`, `get_structure_snapshot_with_lammpslib`, `lammps_run`, `lammps_shutdown`, `calc_molecular_dynamics_npt_with_lammpslib` (existing), `get_potential_by_name` (existing), all imported from `atomistics.calculators.lammps` and `atomistics.calculators.lammps.commands`.

The original notebook (`notebooks/aucu_dist_v9.ipynb`) is left untouched. Read it first with:

```bash
python3 -c "
import json
with open('notebooks/aucu_dist_v9.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    print(f'--- Cell {i} ({cell[\"cell_type\"]}) ---')
    print(''.join(cell['source']))
    print()
"
```

- [ ] **Step 1: Build the new notebook cell-by-cell**

Write a Python script (in the scratchpad directory) that uses `nbformat.v4.new_notebook`, `nbformat.v4.new_code_cell`, `nbformat.v4.new_markdown_cell` to construct the notebook, then `nbformat.write(nb, "notebooks/aucu_dist_v9_atomistics.ipynb")`. Build the cells in this exact order and content (only the diffs from the original are called out; everything else — variable names, values, plotting cells 22-33 — is copied verbatim from the original notebook read above):

**Cell 0 (markdown):** Same intro as the original, with one appended paragraph:

```
---

This version replaces the direct `pylammpsmpi.LammpsASELibrary` calls (session
setup, energy reads, and snapshotting) with the interactive-session primitives
from `atomistics.calculators.lammps`: `lammps_run`/`lammps_shutdown` to
build/tear down a persistent session from a Jinja input template,
`get_energy_pot_with_lammpslib`/`get_energy_pot_average_with_lammpslib` to
read back instantaneous/time-averaged potential energy, and
`get_structure_snapshot_with_lammpslib` to capture cell/positions/velocities.
The Monte Carlo swap logic, cluster-size metrics, and the two-section
comparison are unchanged.
```

**Cell 1 (code) — imports**, replacing the original's imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list
from executorlib import SingleNodeExecutor, split_future
from jinja2 import Template
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from atomistics.calculators.lammps import (
    calc_molecular_dynamics_npt_with_lammpslib,
    get_energy_pot_average_with_lammpslib,
    get_energy_pot_with_lammpslib,
    get_potential_by_name,
    get_structure_snapshot_with_lammpslib,
    lammps_run,
    lammps_shutdown,
)
from atomistics.calculators.lammps.commands import (
    LAMMPS_AVE_ENERGY,
    LAMMPS_ENSEMBLE_NVT,
    LAMMPS_THERMO,
    LAMMPS_THERMO_STYLE,
    LAMMPS_TIMESTEP,
    LAMMPS_VELOCITY,
)
```

**Cell 2 (code):** unchanged from original (`elements_lst`, `structure`, `potential`).

**Cell 3 (code):** unchanged from original (NPT/NVT parameters block).

**Cell 4 (code):** unchanged from original (alloy composition setup).

**Cell 5 (code):** unchanged from original (`elements_lst[1:]`).

**Cell 6 (code) — potential lookup**, replacing the original's `get_potential_dataframe`/manual extraction cell:

```python
df_pot_selected = get_potential_by_name(potential_name=potential)
el_eam_lst = df_pot_selected.Species
el_eam_lst
```

**Cell 7 (code) — template builders**, replacing the original's `LAMMPS_template`/`template.render` cell:

```python
def render_nvt_template(T, seed, thermo, timestep=0.001, velocity_rescale_factor=2.0, dist="gaussian", Tdamp=0.1, include_velocity=True):
    parts = [LAMMPS_THERMO_STYLE, LAMMPS_TIMESTEP, LAMMPS_THERMO]
    if include_velocity:
        parts.append(LAMMPS_VELOCITY)
    parts.append(LAMMPS_ENSEMBLE_NVT)
    return Template("\n".join(parts)).render(
        thermo=thermo,
        timestep=timestep,
        temp=T,
        Tstart=T,
        Tstop=T,
        Tdamp=Tdamp,
        seed=seed,
        dist=dist,
        velocity_rescale_factor=velocity_rescale_factor,
    )


def render_nvt_with_average_template(T, seed, window, timestep=0.001, velocity_rescale_factor=2.0, dist="gaussian", Tdamp=0.1, include_velocity=True, fix_id="avePE"):
    base = render_nvt_template(
        T=T, seed=seed, thermo=window, timestep=timestep,
        velocity_rescale_factor=velocity_rescale_factor, dist=dist, Tdamp=Tdamp,
        include_velocity=include_velocity,
    )
    ave_energy = Template(LAMMPS_AVE_ENERGY).render(
        energy_variable="myPE", fix_id=fix_id, window=window
    )
    return base + "\n" + ave_energy
```

**Cell 8 (code) — helper functions**, replacing the original's cell 9 (`cu_cluster_metrics`, `setup_static_commands`, `init_lammps_session`, `propose_swap`, `apply_swap`, `metropolis_accept`, `get_cluster_metrics_from_lmp`, `make_snapshot`, `restore_snapshot`):

```python
def cu_cluster_metrics(positions, species, cell, pbc, cutoff=cu_cu_cutoff):
    cu_mask = species == 1
    n_cu = int(cu_mask.sum())
    if n_cu == 0:
        return {"n_clusters": 0, "max_cluster_size": 0}
    cu_atoms = Atoms(
        symbols=elements_lst[1:] * n_cu,
        positions=positions[cu_mask],
        cell=cell,
        pbc=pbc,
    )
    i, j = neighbor_list("ij", cu_atoms, cutoff)
    adjacency = coo_matrix((np.ones(len(i)), (i, j)), shape=(n_cu, n_cu))
    n_components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)
    return {"n_clusters": int(n_components), "max_cluster_size": int(sizes.max())}


def propose_swap(species, rng):
    """Pick one random Cu site and one random Au site to swap."""
    cu_sites = np.where(species == 1)[0]
    au_sites = np.where(species == 0)[0]
    i = rng.choice(cu_sites)
    j = rng.choice(au_sites)
    return i, j


def apply_swap(species, i, j):
    """Swap (in place) the species labels at sites i and j; calling this
    twice with the same i, j reverts the swap."""
    species[i], species[j] = species[j], species[i]


def metropolis_accept(delta_e, rng, Temperature):
    return bool(delta_e <= 0 or rng.random() < np.exp(-delta_e / (kB * Temperature)))


def get_cluster_metrics_from_lmp(lmp, species, structure):
    positions = lmp.interactive_positions_getter()
    return cu_cluster_metrics(positions, species, structure.cell, structure.pbc)


def make_snapshot(structure, species, lmp):
    """Capture the current species/cell/positions/velocities as a standalone
    ase.Atoms, using get_structure_snapshot_with_lammpslib for the
    cell/positions/velocities part."""
    struct = structure.copy()
    struct.set_chemical_symbols(np.array(elements_lst)[species].tolist())
    return get_structure_snapshot_with_lammpslib(structure=struct, lmp=lmp)
```

**Cell 9 (code) — `equilibrate_npt`**, replacing the original's custom NPT loop with the existing high-level MD function:

```python
def equilibrate_npt(structure, T, P, n_npt_steps, record_every, Tdamp=0.1, Pdamp=1.0, seed_velocity=12345):
    """Relax cell + positions under NPT, returning (equilibrated_structure,
    df_npt) where equilibrated_structure is an ase.Atoms snapshot (cell,
    positions, velocities) of the relaxed configuration and df_npt is a
    per-record diagnostic trace for checking convergence."""
    output = calc_molecular_dynamics_npt_with_lammpslib(
        structure=structure,
        potential_dataframe=df_pot_selected,
        Tstart=T,
        Tstop=T,
        Tdamp=Tdamp,
        run=n_npt_steps,
        thermo=record_every,
        timestep=0.001,
        Pstart=P,
        Pstop=P,
        Pdamp=Pdamp,
        seed=seed_velocity,
        dist="gaussian",
        velocity_rescale_factor=2.0,
        output_keys=("positions", "cell", "velocities", "energy_pot", "volume"),
    )
    equilibrated = structure.copy()
    equilibrated.set_cell(output["cell"][-1], scale_atoms=False)
    equilibrated.set_positions(output["positions"][-1])
    equilibrated.set_velocities(output["velocities"][-1])
    df_npt = pd.DataFrame(
        {
            "step": record_every * np.arange(1, len(output["energy_pot"]) + 1),
            "e_pot": output["energy_pot"],
            "volume": output["volume"],
        }
    )
    return equilibrated, df_npt
```

**Cell 10 (code) — `equilibrate_structure`**, replacing the original's cell 13:

```python
def equilibrate_structure(structure, n_init_thermalize, lmp_str, species):
    lmp_init = lammps_run(structure=structure, potential_dataframe=df_pot_selected, input_template=lmp_str)
    e_current = get_energy_pot_average_with_lammpslib(lmp=lmp_init, run=n_init_thermalize)
    snapshot = make_snapshot(structure, species, lmp_init)
    metrics = get_cluster_metrics_from_lmp(lmp=lmp_init, species=species, structure=structure)
    lammps_shutdown(lmp_instance=lmp_init)

    return snapshot, {"attempt": -1, "e_pot": e_current, "delta_e": None, "accepted": None, **metrics}, e_current
```

**Cell 11 (code) — `lammps_jump`**, replacing the original's cell 14:

```python
def lammps_jump(structure, species, el_eam_lst, lmp_str, n_md_equil_steps, e_current):
    lmp_attempt = lammps_run(structure=structure, potential_dataframe=df_pot_selected, input_template=lmp_str)
    lmp_attempt.interactive_indices_setter(species, el_eam_lst)
    e_start = get_energy_pot_with_lammpslib(lmp=lmp_attempt, run=0)
    e_new = get_energy_pot_average_with_lammpslib(lmp=lmp_attempt, run=n_md_equil_steps)
    snapshot = make_snapshot(structure, species, lmp_attempt)
    lammps_shutdown(lmp_instance=lmp_attempt)

    delta_e = e_new - e_current
    return delta_e, snapshot, e_start, e_new
```

**Cell 12 (code) — `equilibrate_nvt`**, replacing the original's cell 15:

```python
def equilibrate_nvt(structure, species, lmp_str, n_md_steps, e_delta):
    lmp_md = lammps_run(structure=structure, potential_dataframe=df_pot_selected, input_template=lmp_str)
    e_current = get_energy_pot_with_lammpslib(lmp=lmp_md, run=n_md_steps)  # fixed cadence, win or lose
    snapshot = make_snapshot(structure, species, lmp_md)  # setting the snapshot to the current state after MD step
    metrics = get_cluster_metrics_from_lmp(lmp=lmp_md, species=species, structure=structure)
    lammps_shutdown(lmp_instance=lmp_md)
    return snapshot, {"attempt": -1, "e_pot": e_current, "delta_e": e_delta, "accepted": True, **metrics}, e_current
```

**Cell 13 (code) — `classical_mdmc`**, replacing the original's cell 16 (same docstring and control flow; only the LAMMPS calls change):

```python
def classical_mdmc(structure, species_idx, el_eam_lst, seed, n_init_thermalize, n_mc_attempts, n_total, n_md_steps, mc_temperature, lmp_str, label="Section A"):
    """Instant evaluation: each trial swap is judged with a cheap `run 0`
    against the current frozen configuration. Regardless of accept/reject,
    MD then runs forward by a fixed `attempt_interval_steps` before the next
    attempt, so the attempt cadence (attempts per unit of MD time) stays
    constant no matter how many attempts (`n_attempts`) the run consists
    of."""
    rng = np.random.default_rng(seed)
    species = species_idx.copy()
    results = []

    lmp = lammps_run(structure=structure, potential_dataframe=df_pot_selected, input_template=lmp_str)
    e_current = get_energy_pot_with_lammpslib(lmp=lmp, run=n_init_thermalize)
    metrics = get_cluster_metrics_from_lmp(lmp=lmp, species=species, structure=structure)
    results.append({"attempt": -1, "e_pot": e_current, "delta_e": None, "accepted": None, **metrics})

    for _ in tqdm(range(n_total)):
        for attempt in range(n_mc_attempts):
            i, j = propose_swap(species, rng)
            apply_swap(species, i, j)
            lmp.interactive_indices_setter(species, el_eam_lst)
            delta_e = get_energy_pot_with_lammpslib(lmp=lmp, run=0) - e_current
            accept = metropolis_accept(delta_e, rng, Temperature=mc_temperature)

            if not accept:
                apply_swap(species, i, j)  # revert
                lmp.interactive_indices_setter(species, el_eam_lst)
            else:
                break  # if accepted, skip remaining attempts and move on to MD step

        e_current = get_energy_pot_with_lammpslib(lmp=lmp, run=n_md_steps)
        metrics = get_cluster_metrics_from_lmp(lmp=lmp, species=species, structure=structure)
        results.append({"attempt": attempt, "e_pot": e_current, "delta_e": delta_e, "accepted": accept, **metrics})

    lammps_shutdown(lmp_instance=lmp)

    return pd.DataFrame(results)
```

**Cell 14 (code) — `parallel_mdmc`**, replacing the original's cell 17 (same docstring and control flow; `lmp_str_resume` is a new parameter used for every session rebuild after the first):

```python
def parallel_mdmc(executor, structure, species_idx, el_eam_lst, seed, n_init_thermalize, n_mc_attempts, n_total, n_md_steps, n_md_equil_steps, mc_temperature, lmp_str, lmp_str_resume, label="Section B"):
    """Relax-then-evaluate: each trial swap runs `attempt_interval_steps` of
    real MD, and the accept/reject decision uses the post-window averaged PE
    (LAMMPS-native fix ave/time, f_avePE) rather than the instantaneous
    value. Accepted swaps simply continue forward; rejected swaps revert
    only the species labels -- positions/velocities are *not* rewound -- so
    every attempt costs exactly `attempt_interval_steps` of MD either way,
    matching Section A's cadence."""
    rng = np.random.default_rng(seed)
    species = species_idx.copy()
    results = []
    e_start_lst, e_new_lst = [], []

    future = executor.submit(
        equilibrate_structure,
        structure=structure,
        n_init_thermalize=n_init_thermalize,
        lmp_str=lmp_str,
        species=species,
    )
    snapshot, result, e_current = future.result()
    results.append(result)

    for _ in tqdm(range(n_total)):
        delta_e_dict = {}
        species_dict = {}
        future_dict = {}
        for attempt in range(n_mc_attempts):
            species_copy = species.copy()
            i, j = propose_swap(species_copy, rng)
            apply_swap(species_copy, i, j)
            future = executor.submit(
                lammps_jump,
                structure=snapshot,
                species=species_copy,
                el_eam_lst=el_eam_lst,
                lmp_str=lmp_str_resume,
                n_md_equil_steps=n_md_equil_steps,
                e_current=e_current,
            )
            future_dict[attempt] = future
            species_dict[attempt] = species_copy

        e_start_tmp_lst, e_new_tmp_lst = [], []
        for a, f in future_dict.items():
            delta_e, snapshot_step, e_start, e_new = f.result()
            delta_e_dict[delta_e] = snapshot_step
            species_dict[delta_e] = species_dict[a]
            e_start_tmp_lst.append(e_start)
            e_new_tmp_lst.append(e_new)

        e_delta = min(delta_e_dict.keys())
        if e_delta < 0:
            snapshot = delta_e_dict[e_delta]
            species = species_dict[e_delta]
            accepted = True
        else:
            accepted = False
        future = executor.submit(
            equilibrate_nvt, structure=snapshot, species=species, lmp_str=lmp_str_resume, n_md_steps=n_md_steps, e_delta=e_delta)
        snapshot, result, e_current = future.result()
        result["accepted"] = accepted
        results.append(result)
        e_start_lst.append(e_start_tmp_lst)
        e_new_lst.append(e_new_tmp_lst)
    return pd.DataFrame(results), e_start_lst, e_new_lst
```

**Cell 15 (markdown):** unchanged from original (`# MD=50 K`).

**Cell 16 (code):** unchanged from original (`T = 200.0`).

**Cell 17 (code) — template rendering**, replacing the original's `lmp_str = template.render(...)` cell:

```python
lmp_str = render_nvt_with_average_template(
    T=T,
    seed=12345,
    window=n_nvt_equilibrate_steps,
    timestep=0.001,
    velocity_rescale_factor=2.0,
    dist="gaussian",
    Tdamp=0.1,
    include_velocity=True,
)
lmp_str_resume = render_nvt_with_average_template(
    T=T,
    seed=12345,
    window=n_nvt_equilibrate_steps,
    timestep=0.001,
    velocity_rescale_factor=2.0,
    dist="gaussian",
    Tdamp=0.1,
    include_velocity=False,
)
lmp_str
```

**Cell 18 (code) — main execution**, replacing the original's cell 21 (only the `equilibrate_npt`/`classical_mdmc`/`parallel_mdmc` call arguments change — no `el_eam_lst` for `equilibrate_npt`, and `parallel_mdmc` gets `lmp_str_resume`):

```python
with SingleNodeExecutor(hostname_localhost=True, block_allocation=True, max_workers=4) as exe:
    future_equilibrate = exe.submit(
        equilibrate_npt,
        structure=structure,
        T=T,
        P=P,
        n_npt_steps=n_npt_steps,
        record_every=n_npt_record_every,
    )
    equilibrated_structure_future, df_npt_future = split_future(future=future_equilibrate, n=2)
    df_results_a_future = exe.submit(
        classical_mdmc,
        structure=equilibrated_structure_future,
        species_idx=species_idx,
        el_eam_lst=el_eam_lst,
        seed=7,
        n_init_thermalize=n_nvt_init_steps,
        n_mc_attempts=n_mc_trys,
        n_total=n_mdmc_steps,
        n_md_steps=n_nvt_md_between_mc,
        label="Section A",
        mc_temperature=T,
        lmp_str=lmp_str,
    )
    df_results_b, e_start_lst, e_new_lst = parallel_mdmc(
        executor=exe,
        structure=equilibrated_structure_future,
        species_idx=species_idx,
        el_eam_lst=el_eam_lst,
        seed=7,
        n_init_thermalize=n_nvt_init_steps,
        n_mc_attempts=n_mc_trys,
        n_total=n_mdmc_steps,
        n_md_steps=n_nvt_md_between_mc,
        n_md_equil_steps=n_nvt_equilibrate_steps,
        label="Section B",
        mc_temperature=10.0,
        lmp_str=lmp_str,
        lmp_str_resume=lmp_str_resume,
    )

    df_results_a = df_results_a_future.result()
    equilibrated_structure, df_npt = future_equilibrate.result()


print("original volume:  ", structure.get_volume())
print("equilibrated volume:", equilibrated_structure.get_volume())
print("volume change:     ", equilibrated_structure.get_volume() - structure.get_volume(),
      f"({100 * (equilibrated_structure.get_volume() / structure.get_volume() - 1):.3f}%)")
print()
print("original cell:\n", structure.cell.array)
print("equilibrated cell:\n", equilibrated_structure.cell.array)
```

**Cells 19-30 (markdown/code):** copy verbatim from the original notebook's cells 22-33 (all the plotting/analysis cells) — no LAMMPS-specific code in any of them, so no changes needed.

- [ ] **Step 2: Sanity-check the notebook parses and every referenced name resolves**

Run:

```bash
jupyter nbconvert --to script --stdout notebooks/aucu_dist_v9_atomistics.ipynb | python3 -c "import ast, sys; ast.parse(sys.stdin.read())"
```

Expected: no output, exit code 0 (confirms valid Python syntax across all cells).

- [ ] **Step 3: Smoke-test the ported MD/MC logic at tiny scale**

Runtime cost warning: the full notebook (`n_npt_steps=5000`, `n_mdmc_steps=200`, `n_mc_trys=100`, `n_nvt_equilibrate_steps=1000`) involves on the order of 10^5-10^7 LAMMPS timesteps and, in Section B, tens of thousands of individual session rebuilds through `executorlib` — this is expensive by construction (inherited unchanged from the original notebook) and is not meant to be executed as part of this task. Do not attempt to execute the full notebook in this task.

Instead, write and run a standalone throwaway script (not saved, e.g. via `python3 -c` or a scratchpad file) that imports the notebook's functions via `jupyter nbconvert --to script` and exercises `classical_mdmc` and `parallel_mdmc` at trivial scale, to catch integration bugs (wrong argument names, wrong template composition, snapshot species not propagating) before declaring the port done:

```bash
jupyter nbconvert --to script --stdout notebooks/aucu_dist_v9_atomistics.ipynb > /tmp/aucu_dist_v9_atomistics_smoke.py
python3 - <<'EOF'
import re
with open("/tmp/aucu_dist_v9_atomistics_smoke.py") as f:
    src = f.read()
# Drop the main-execution and plotting cells (everything from the "with SingleNodeExecutor" block onward)
src = src.split("with SingleNodeExecutor(hostname_localhost=True, block_allocation=True, max_workers=4) as exe:")[0]
with open("/tmp/aucu_dist_v9_atomistics_smoke.py", "w") as f:
    f.write(src)
EOF
python3 - <<'EOF'
import sys
sys.path.insert(0, "/tmp")
exec(open("/tmp/aucu_dist_v9_atomistics_smoke.py").read())

from executorlib import SingleNodeExecutor, split_future

tiny_structure = structure[:32].copy()  # small enough to run fast
tiny_structure.set_chemical_symbols(["Au"] * 26 + ["Cu"] * 6)
tiny_species_idx = np.array([0] * 26 + [1] * 6)

equilibrated, df_npt = equilibrate_npt(
    structure=tiny_structure, T=200.0, P=0.0, n_npt_steps=20, record_every=10,
)
print("NPT smoke ok:", equilibrated.get_volume(), len(df_npt))

lmp_str_smoke = render_nvt_with_average_template(T=200.0, seed=1, window=5, include_velocity=True)
lmp_str_smoke_resume = render_nvt_with_average_template(T=200.0, seed=1, window=5, include_velocity=False)

df_a = classical_mdmc(
    structure=equilibrated, species_idx=tiny_species_idx, el_eam_lst=el_eam_lst, seed=1,
    n_init_thermalize=5, n_mc_attempts=2, n_total=2, n_md_steps=5, mc_temperature=200.0,
    lmp_str=lmp_str_smoke,
)
print("Section A smoke ok:\n", df_a)

with SingleNodeExecutor(hostname_localhost=True, block_allocation=True, max_workers=2) as exe:
    df_b, e_start_lst, e_new_lst = parallel_mdmc(
        executor=exe, structure=equilibrated, species_idx=tiny_species_idx, el_eam_lst=el_eam_lst, seed=1,
        n_init_thermalize=5, n_mc_attempts=2, n_total=2, n_md_steps=5, n_md_equil_steps=5,
        mc_temperature=200.0, lmp_str=lmp_str_smoke, lmp_str_resume=lmp_str_smoke_resume,
    )
print("Section B smoke ok:\n", df_b)
EOF
```

Expected: both "Section A smoke ok" and "Section B smoke ok" print small `DataFrame`s with finite `e_pot` values and no traceback. If it fails, fix the specific cell in `notebooks/aucu_dist_v9_atomistics.ipynb` that the traceback points to (most likely culprits: an argument name mismatch between a cell and `classical_mdmc`/`parallel_mdmc`, or `el_eam_lst` missing from a call site) and re-run this step.

- [ ] **Step 4: Delete the throwaway smoke artifacts**

```bash
rm -f /tmp/aucu_dist_v9_atomistics_smoke.py
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/aucu_dist_v9_atomistics.ipynb
git commit -m "Add aucu_dist_v9_atomistics.ipynb: port hybrid MD/MC notebook onto new LAMMPS session primitives"
```

Note for whoever runs this notebook for real: the full run (same parameters as `aucu_dist_v9.ipynb`) has not been executed end-to-end as part of this task, for the runtime-cost reason given in Step 3 — only a reduced-parameter smoke variant was verified. Running it for real is the same cost as running the original notebook.

---

## Plan Self-Review Notes

- **Spec coverage:** `LAMMPS_AVE_ENERGY` (Task 2), `get_energy_pot_with_lammpslib` (Task 1), `get_energy_pot_average_with_lammpslib` (Task 2), `get_structure_snapshot_with_lammpslib` (Task 3), public export of all three plus `lammps_run`/`lammps_shutdown` (Task 4), test file (Tasks 1-4), updated notebook with original left untouched (Task 5) — every design-doc item has a task.
- **Type consistency:** `get_energy_pot_with_lammpslib(lmp, run=0)`, `get_energy_pot_average_with_lammpslib(lmp, run, fix_id="avePE")`, `get_structure_snapshot_with_lammpslib(structure, lmp)` are used with identical signatures across Tasks 1-3, the Task 4 export lists, and every notebook cell in Task 5.
- **Notebook function signatures:** `equilibrate_structure`, `lammps_jump`, `equilibrate_nvt`, `classical_mdmc`, `parallel_mdmc` signatures are consistent between their Task 5 definitions and their call sites in cell 18 — cross-checked `el_eam_lst` is present only where `interactive_indices_setter` is actually called (`lammps_jump`, `classical_mdmc`, `parallel_mdmc`), and `lmp_str_resume` is threaded through `parallel_mdmc` to both of its internal `lammps_jump`/`equilibrate_nvt` submissions.
