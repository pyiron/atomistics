LAMMPS_THERMO_STYLE = """\
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g"""


LAMMPS_THERMO = "thermo {{thermo}}"


LAMMPS_RUN = "run {{run}}"


LAMMPS_MINIMIZE = """\
min_style {{min_style}}
minimize {{etol}} {{ftol}} {{maxiter}} {{maxeval}}"""


LAMMPS_MINIMIZE_VOLUME = "fix ensemble all box/relax iso 0.0"


LAMMPS_TIMESTEP = "timestep {{timestep}}"


LAMMPS_VELOCITY = "velocity all create $(2 * {{ temp }}) {{seed}} dist {{dist}}"


LAMMPS_ENSEMBLE_NPT = "fix ensemble all npt temp {{Tstart}} {{Tstop}} {{Tdamp}} iso {{Pstart}} {{Pstop}} {{Pdamp}}"


LAMMPS_ENSEMBLE_NPH = "fix ensemble all nph iso {{Pstart}} {{Pstop}} {{Pdamp}}"


LAMMPS_ENSEMBLE_NVT = "fix ensemble all nvt temp {{Tstart}} {{Tstop}} {{Tdamp}}"


LAMMPS_LANGEVIN = "fix ensemble all langevin {{Tstart}} {{Tstop}} {{Tdamp}} {{seed}}"


LAMMPS_NVE = "fix integration all nve"
