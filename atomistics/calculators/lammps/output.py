from atomistics.shared.output import OutputMolecularDynamics, OutputStatic


class LammpsOutput(OutputMolecularDynamics, OutputStatic):
    def __init__(self, lmp):
        self._lmp = lmp

    @property
    def forces(self):
        return self._lmp.interactive_forces_getter()

    @property
    def energy(self):
        return self._lmp.interactive_energy_pot_getter()

    @property
    def stress(self):
        return self._lmp.interactive_pressures_getter()

    @property
    def volume(self):
        return self._lmp.interactive_volume_getter()

    @property
    def positions(self):
        return self._lmp.interactive_positions_getter()

    @property
    def cell(self):
        return self._lmp.interactive_cells_getter()

    @property
    def temperature(self):
        return self._lmp.interactive_temperatures_getter()

    @property
    def energy_pot(self):
        return self._lmp.interactive_energy_pot_getter()

    @property
    def energy_tot(self):
        return self._lmp.interactive_energy_tot_getter()

    @property
    def pressure(self):
        return self._lmp.interactive_pressures_getter()

    @property
    def velocities(self):
        return self._lmp.interactive_velocities_getter()
