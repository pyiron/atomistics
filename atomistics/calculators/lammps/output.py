from atomistics.shared.output import (
    OutputStatic,
    OutputMolecularDynamics,
)
from pylammpsmpi import LammpsASELibrary


LammpsOutputStatic = OutputStatic(
    forces=LammpsASELibrary.interactive_forces_getter,
    energy=LammpsASELibrary.interactive_energy_pot_getter,
    stress=LammpsASELibrary.interactive_pressures_getter,
    volume=LammpsASELibrary.interactive_volume_getter,
)
LammpsOutputMolecularDynamics = OutputMolecularDynamics(
    positions=LammpsASELibrary.interactive_positions_getter,
    cell=LammpsASELibrary.interactive_cells_getter,
    forces=LammpsASELibrary.interactive_forces_getter,
    temperature=LammpsASELibrary.interactive_temperatures_getter,
    energy_pot=LammpsASELibrary.interactive_energy_pot_getter,
    energy_tot=LammpsASELibrary.interactive_energy_tot_getter,
    pressure=LammpsASELibrary.interactive_pressures_getter,
    velocities=LammpsASELibrary.interactive_velocities_getter,
    volume=LammpsASELibrary.interactive_volume_getter,
)
