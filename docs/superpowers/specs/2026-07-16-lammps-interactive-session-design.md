# LAMMPS interactive-session primitives for hybrid MD/MC

## Problem

`notebooks/aucu_dist_v9.ipynb` implements a hybrid molecular-dynamics /
Monte-Carlo (MD/MC) atom-swap simulation for an Au-Cu alloy directly on top of
`pylammpsmpi.LammpsASELibrary`. It drives a persistent LAMMPS session by hand:

- running fixed-length MD segments and reading back the potential energy,
  either instantaneously or as a LAMMPS-native time average read through the
  private `lmp._interactive_library.extract_fix(...)` attribute,
- swapping atom species mid-session via `interactive_indices_setter`,
- snapshotting the full session state (cell + positions + velocities) into a
  standalone `ase.Atoms` object so it can be restored into a fresh session
  later (there is no velocity setter, so a full session rebuild is the only
  way to restore velocities).

None of this has a home in `atomistics.calculators.lammps` today. Every
existing public function there (`calc_static_with_lammpslib`,
`calc_molecular_dynamics_nvt_with_lammpslib`, etc.) is a one-shot
"build session → run → close" call; there is no supported way to hold a
session open, advance it in small increments, and read back energy or a full
state snapshot along the way.

## Scope

Add low-level session primitives only. The MC-swap loop logic (propose /
accept / reject, the two loop variants used in the notebook — sequential
"instant evaluate" and parallel "relax-then-evaluate" via `executorlib`),
Cu-cluster-size metrics, and other alloy-specific code stay in the notebook.
The goal is for the notebook to stop reaching into `pylammpsmpi` and
duplicating boilerplate, not to encode the MD/MC algorithm itself into
`atomistics`.

## New functions (`atomistics/calculators/lammps/libcalculator.py`)

All three follow the package's existing `_with_lammpslib` naming and are
exported from `atomistics.calculators.lammps` and `atomistics.calculators`
(added to the `lammps_functions` list / try-except-import block in both
`__init__.py` files, matching the existing pattern).

### `get_energy_pot_with_lammpslib(lmp, run: int = 0) -> float`

Advances the given live session by `run` timesteps (`interactive_lib_command(f"run {run}")`)
and returns the instantaneous potential energy
(`interactive_energy_pot_getter()`). Direct replacement for the notebook's
`get_energy_pot`.

### `get_energy_pot_average_with_lammpslib(lmp, run: int, fix_id: str = "avePE") -> float`

Advances the session by `run` timesteps, then reads the time-averaged
potential energy from a previously-defined `fix ave/time` via
`lmp._interactive_library.extract_fix(fix_id, 0, 0)` — mirroring the
notebook's `get_energy_avg`, including its reliance on the private
`pylammpsmpi` attribute (per user decision: wrap the private attribute rather
than reimplementing the average in Python).

Docstring must note:

- This only works when the session was created with `cores=1` and no
  `executor` argument, so `_interactive_library` is the raw `lammps.lammps`
  object (the branch pylammpsmpi takes in that case). Sessions created via an
  `executorlib` executor or with `cores > 1` use a different wrapper object
  and are not supported by this function.
- `run` should equal (or be an integer multiple of) the window size the
  `fix ave/time` was configured with (see `LAMMPS_AVE_ENERGY` below),
  otherwise the extracted value reflects a stale or incomplete averaging
  window.

### `get_structure_snapshot_with_lammpslib(structure, lmp) -> Atoms`

Returns a copy of `structure` with cell (`interactive_cells_getter`,
`scale_atoms=False`), positions (`interactive_positions_getter`), and
velocities (`interactive_velocities_getter`) overwritten from the live
session. Chemical symbols are left untouched on the passed-in `structure` —
callers that changed species mid-session (e.g. after an MC swap) set the new
symbols on `structure` themselves before calling this function. Replacement
for the notebook's `make_snapshot` (minus its alloy-specific symbol
assignment, which stays in the notebook).

## New command template (`atomistics/calculators/lammps/commands.py`)

```python
LAMMPS_AVE_ENERGY = """\
variable {{energy_variable}} equal pe
fix {{fix_id}} all ave/time 1 {{window}} {{window}} v_{{energy_variable}}"""
```

Follows the existing Jinja-template-constant pattern used by
`LAMMPS_THERMO`, `LAMMPS_VELOCITY`, `LAMMPS_ENSEMBLE_NVT`, etc., so the
notebook composes its NVT + energy-averaging input template the same way the
existing `calc_molecular_dynamics_*_with_lammpslib` functions compose theirs.

## Exposed session primitives

`lammps_run` and `lammps_shutdown` (already implemented in `helpers.py` and
used internally throughout `libcalculator.py`, `melting.py`) are exported
publicly (added to `atomistics.calculators.lammps.__init__` and
`atomistics.calculators.__init__`). The notebook needs to build and tear down
a persistent session using a custom input template — exactly what these two
functions already do; they are just not currently part of the public API.

## What stays in the notebook

- Swap proposal / accept-reject logic (`propose_swap`, `apply_swap`,
  `metropolis_accept`) — generic MC, not LAMMPS-specific, but small enough
  that it isn't worth extracting.
- The two MD/MC loop variants (`classical_mdmc` sequential "instant
  evaluate", `parallel_mdmc` parallel "relax-then-evaluate" via
  `executorlib`), now built on the three new functions instead of raw
  `pylammpsmpi` calls.
- Cu-cluster-size metrics (`cu_cluster_metrics`) — alloy-specific, uses
  `structuretoolkit`/`ase.neighborlist`, unrelated to the LAMMPS interface.
- NPT pre-equilibration: the notebook's custom `equilibrate_npt` is deleted
  outright rather than ported, since it is already fully covered by the
  existing `calc_molecular_dynamics_npt_with_lammpslib` — its output dict
  already contains `positions`/`cell`/`velocities`/`volume`/`energy_pot`
  arrays at the requested `thermo` cadence, so the final relaxed state and
  the convergence trace both come for free from one existing call.

## Tests

New file `tests/test_lammps_interactive_session.py`, following the existing
`unittest.TestCase` + `@unittest.skipIf(skip_lammps_test, ...)` +
`get_potential_by_name(..., resource_path=".../static/lammps")` pattern used
throughout `tests/`. Covers:

- `get_energy_pot_with_lammpslib` after a run matches a direct
  `interactive_energy_pot_getter()` call on the same session.
- `get_energy_pot_average_with_lammpslib` returns a finite value that need
  not equal the instantaneous energy (thermal noise), using a session built
  with the new `LAMMPS_AVE_ENERGY` template.
- `get_structure_snapshot_with_lammpslib` returns an `Atoms` object whose
  cell/positions/velocities match the live session, and that snapshot can be
  fed back into `lammps_run` to resume a session (round-trip).

## Notebook

New file `notebooks/aucu_dist_v9_atomistics.ipynb`: same simulation, same
structure and cell sections as `aucu_dist_v9.ipynb`, with the
`pylammpsmpi`-boilerplate helper functions (`init_lammps_session`,
`setup_static_commands`, `get_energy_pot`, `get_energy_avg`, `make_snapshot`,
`equilibrate_npt`) replaced by calls into the three new `atomistics`
functions plus the existing `calc_molecular_dynamics_npt_with_lammpslib`.
The original notebook is left untouched.
