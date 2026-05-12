import os
import pandas as pd
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from ase.build import bulk

try:
    from atomistics.calculators.lammps.libcalculator import (
        calc_molecular_dynamics_nvt_with_lammpslib,
        calc_molecular_dynamics_npt_with_lammpslib,
        calc_molecular_dynamics_nph_with_lammpslib,
        calc_molecular_dynamics_langevin_with_lammpslib,
        calc_molecular_dynamics_thermal_expansion_with_lammpslib,
    )
    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestLammpsVelocityRescaleMock(unittest.TestCase):
    def setUp(self):
        self.structure = bulk("Al", cubic=True)
        self.potential_dataframe = pd.DataFrame({
            "Config": [["pair_style eam/alloy", "pair_coeff * * Al.eam.alloy Al"]],
            "Species": [["Al"]],
            "Model": ["eam/alloy"],
            "Name": ["Al_Mishin"],
            "Filename": [["Al.eam.alloy"]]
        })

    def _setup_mock_lmp(self, MockLammpsASELibrary):
        mock_lmp = MockLammpsASELibrary.return_value
        mock_lmp.interactive_positions_getter = MagicMock(return_value=self.structure.positions)
        mock_lmp.interactive_cells_getter = MagicMock(return_value=self.structure.cell)
        mock_lmp.interactive_forces_getter = MagicMock(return_value=self.structure.positions * 0)
        mock_lmp.interactive_temperatures_getter = MagicMock(return_value=100.0)
        mock_lmp.interactive_energy_pot_getter = MagicMock(return_value=-1.0)
        mock_lmp.interactive_energy_tot_getter = MagicMock(return_value=-0.5)
        mock_lmp.interactive_pressures_getter = MagicMock(return_value=np.zeros((3,3)))
        mock_lmp.interactive_velocities_getter = MagicMock(return_value=self.structure.positions * 0)
        mock_lmp.interactive_volume_getter = MagicMock(return_value=self.structure.get_volume())
        return mock_lmp

    @patch("atomistics.calculators.lammps.helpers.validate_potential_dataframe")
    @patch("atomistics.calculators.lammps.helpers.LammpsASELibrary")
    def test_nvt_velocity_rescale(self, MockLammpsASELibrary, MockValidate):
        MockValidate.return_value = self.potential_dataframe
        mock_lmp = self._setup_mock_lmp(MockLammpsASELibrary)

        # Default factor
        calc_molecular_dynamics_nvt_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            Tstart=100.0,
            velocity_rescale_factor=2.0
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertTrue(any("velocity all create $(2.0 * 100.0)" in c for c in calls))

        mock_lmp.interactive_lib_command.reset_mock()
        # Custom factor
        calc_molecular_dynamics_nvt_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            Tstart=100.0,
            velocity_rescale_factor=3.5
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertTrue(any("velocity all create $(3.5 * 100.0)" in c for c in calls))

        mock_lmp.interactive_lib_command.reset_mock()
        # None
        calc_molecular_dynamics_nvt_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            velocity_rescale_factor=None
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertFalse(any("velocity all create" in c for c in calls))

    @patch("atomistics.calculators.lammps.helpers.validate_potential_dataframe")
    @patch("atomistics.calculators.lammps.helpers.LammpsASELibrary")
    def test_npt_velocity_rescale(self, MockLammpsASELibrary, MockValidate):
        MockValidate.return_value = self.potential_dataframe
        mock_lmp = self._setup_mock_lmp(MockLammpsASELibrary)

        calc_molecular_dynamics_npt_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            Tstart=100.0,
            velocity_rescale_factor=2.5
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertTrue(any("velocity all create $(2.5 * 100.0)" in c for c in calls))

        mock_lmp.interactive_lib_command.reset_mock()
        calc_molecular_dynamics_npt_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            velocity_rescale_factor=None
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertFalse(any("velocity all create" in c for c in calls))

    @patch("atomistics.calculators.lammps.helpers.validate_potential_dataframe")
    @patch("atomistics.calculators.lammps.helpers.LammpsASELibrary")
    def test_nph_velocity_rescale(self, MockLammpsASELibrary, MockValidate):
        MockValidate.return_value = self.potential_dataframe
        mock_lmp = self._setup_mock_lmp(MockLammpsASELibrary)

        calc_molecular_dynamics_nph_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            Tstart=100.0,
            velocity_rescale_factor=1.5
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertTrue(any("velocity all create $(1.5 * 100.0)" in c for c in calls))

        mock_lmp.interactive_lib_command.reset_mock()
        calc_molecular_dynamics_nph_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            velocity_rescale_factor=None
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertFalse(any("velocity all create" in c for c in calls))

    @patch("atomistics.calculators.lammps.helpers.validate_potential_dataframe")
    @patch("atomistics.calculators.lammps.helpers.LammpsASELibrary")
    def test_langevin_velocity_rescale(self, MockLammpsASELibrary, MockValidate):
        MockValidate.return_value = self.potential_dataframe
        mock_lmp = self._setup_mock_lmp(MockLammpsASELibrary)

        calc_molecular_dynamics_langevin_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            Tstart=100.0,
            velocity_rescale_factor=4.0
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertTrue(any("velocity all create $(4.0 * 100.0)" in c for c in calls))

        mock_lmp.interactive_lib_command.reset_mock()
        calc_molecular_dynamics_langevin_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            run=10, thermo=10,
            velocity_rescale_factor=None
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        self.assertFalse(any("velocity all create" in c for c in calls))

    @patch("atomistics.calculators.lammps.helpers.validate_potential_dataframe")
    @patch("atomistics.calculators.lammps.helpers.LammpsASELibrary")
    def test_thermal_expansion_velocity_rescale(self, MockLammpsASELibrary, MockValidate):
        MockValidate.return_value = self.potential_dataframe
        mock_lmp = self._setup_mock_lmp(MockLammpsASELibrary)

        calc_molecular_dynamics_thermal_expansion_with_lammpslib(
            structure=self.structure,
            potential_dataframe=self.potential_dataframe,
            Tstart=50, Tstop=100, Tstep=50,
            run=10, thermo=10,
            velocity_rescale_factor=3.0
        )
        calls = [call[0][0] for call in mock_lmp.interactive_lib_command.call_args_list]
        # Thermal expansion uses Tstart for velocity creation
        # We check the rendered string for the velocity command
        velocity_call = [c for c in calls if "velocity all create" in c]
        self.assertTrue(len(velocity_call) > 0, "Velocity command not found in calls")
        # Floating point representation in Jinja2 might vary, but for 3.0 and 50.0 it should be straightforward
        # Actually it might be 3.0 * 50.0 or 150.0 depending on how it's handled.
        # Wait, Jinja2 renders $( {{velocity_rescale_factor}} * {{ temp }} )
        # So it should be $(3.0 * 50)
        self.assertTrue(any("velocity all create $(3.0 * 50)" in c for c in calls), f"Calls were: {calls}")

if __name__ == "__main__":
    unittest.main()
