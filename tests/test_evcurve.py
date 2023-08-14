import os
import unittest
import pyiron_lammps as pyr


def validate_elastic_constants(fit_dict):
    print(fit_dict)
    return [
        fit_dict['b_prime_eq'] > 3.3,
        fit_dict['b_prime_eq'] < 3.9,
        fit_dict['bulkmodul_eq'] > 183,
        fit_dict['bulkmodul_eq'] < 187,
        fit_dict['energy_eq'] > -453.9,
        fit_dict['energy_eq'] < -453.6,
        fit_dict['volume_eq'] > 1208,
        fit_dict['volume_eq'] < 1212,
    ]


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        count_lst = [22, 22, 22, 21, 21]
        element_lst = ["Fe", "Ni", "Cr", "Co", "Cu"]
        potential = '2021--Deluigi-O-R--Fe-Ni-Cr-Co-Cu--LAMMPS--ipr1'
        resource_path = os.path.join(os.path.dirname(__file__), "static")

        # Generate SQS Structure
        structure = pyr.generate_sqs_structure(
            structure_template=pyr.get_ase_bulk("Al", cubic=True).repeat([3, 3, 3]),
            element_lst=element_lst,
            count_lst=count_lst
        )[0]

        # Select potential
        df_pot = pyr.get_potential_dataframe(
            structure=structure,
            resource_path=resource_path
        )

        # Assign variable
        cls.df_pot_selected = df_pot[df_pot.Name == potential].iloc[0]
        cls.structure = structure
        cls.resource_path = resource_path
        cls.potential = potential
        cls.count_lst = count_lst

    def test_structure(self):
        self.assertEqual(len(self.structure), sum(self.count_lst))

    def test_example_evcurve_with_one_executable(self):
        # Initialize Engine
        lmp = pyr.get_lammps_engine()

        # Optimize Structure
        structure_opt = pyr.optimize_structure(
            lmp=lmp,
            structure=self.structure.copy(),
            potential_dataframe=self.df_pot_selected
        )
        self.assertEqual(len(structure_opt), sum(self.count_lst))

        # Calculate Elastic Constants
        ev_curve_fit_dict = pyr.calculate_energy_volume_curve(
            structure=structure_opt,
            potential_dataframe=self.df_pot_selected,
            num_points=11,
            fit_type="polynomial",
            fit_order=3,
            vol_range=0.1,
            axes=["x", "y", "z"],
            strains=None,
        )
        self.assertTrue(all(validate_elastic_constants(fit_dict=ev_curve_fit_dict)))

        # Finalize
        lmp.close()

    def test_example_evcurve_with_separate_executable(self):
        # Optimize Structure
        structure_opt = pyr.optimize_structure(
            structure=self.structure.copy(),
            potential_dataframe=self.df_pot_selected
        )
        self.assertEqual(len(structure_opt), sum(self.count_lst))

        # Calculate Elastic Constants
        ev_curve_fit_dict = pyr.calculate_energy_volume_curve(
            structure=structure_opt,
            potential_dataframe=self.df_pot_selected,
            num_points=11,
            fit_type="polynomial",
            fit_order=3,
            vol_range=0.1,
            axes=["x", "y", "z"],
            strains=None,
        )
        self.assertTrue(all(validate_elastic_constants(fit_dict=ev_curve_fit_dict)))

    def test_example_evcurve_with_statement(self):
        with pyr.get_lammps_engine() as lmp:
            # Optimize Structure
            structure_opt = pyr.optimize_structure(
                lmp=lmp,
                structure=self.structure.copy(),
                potential_dataframe=self.df_pot_selected
            )

            # Calculate Elastic Constants
            ev_curve_fit_dict = pyr.calculate_energy_volume_curve(
                lmp=lmp,
                structure=structure_opt,
                potential_dataframe=self.df_pot_selected,
                num_points=11,
                fit_type="polynomial",
                fit_order=3,
                vol_range=0.1,
                axes=["x", "y", "z"],
                strains=None,
            )

        self.assertEqual(len(structure_opt), sum(self.count_lst))
        self.assertTrue(all(validate_elastic_constants(fit_dict=ev_curve_fit_dict)))

    def test_example_evcurve_with_minimization(self):
        with pyr.get_lammps_engine() as lmp:
            ev_curve_fit_dict = pyr.calculate_energy_volume_curve_with_minimization(
                lmp=lmp,
                structure=self.structure.copy(),
                potential_dataframe=self.df_pot_selected,
                num_points=11,
                fit_type="polynomial",
                fit_order=3,
                vol_range=0.1,
                axes=["x", "y", "z"],
                strains=None,
            )

        self.assertTrue(all(validate_elastic_constants(fit_dict=ev_curve_fit_dict)))


if __name__ == '__main__':
    unittest.main()
