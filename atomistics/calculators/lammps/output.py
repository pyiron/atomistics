from atomistics.shared.output import OutputMolecularDynamics, OutputStatic


class LammpsOutput(OutputMolecularDynamics, OutputStatic):
    def __init__(self, lmp):
        self._lmp = lmp

    def forces(self):
        return self._lmp.interactive_forces_getter()

    def energy(self):
        return self._lmp.interactive_energy_pot_getter()

    def stress(self):
        return self._lmp.interactive_pressures_getter()

    def volume(self):
        return self._lmp.interactive_volume_getter()

    def positions(self):
        return self._lmp.interactive_positions_getter()

    def cell(self):
        return self._lmp.interactive_cells_getter()

    def temperature(self):
        return self._lmp.interactive_temperatures_getter()

    def energy_pot(self):
        return self._lmp.interactive_energy_pot_getter()

    def energy_tot(self):
        return self._lmp.interactive_energy_tot_getter()

    def pressure(self):
        return self._lmp.interactive_pressures_getter()

    def velocities(self):
        return self._lmp.interactive_velocities_getter()
