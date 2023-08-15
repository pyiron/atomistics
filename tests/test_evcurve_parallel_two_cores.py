import os
import unittest
from ase.build import bulk
import pyiron_lammps as pyr
import structuretoolkit as stk


def validate_fitdict(fit_dict):
    lst = [
        fit_dict['b_prime_eq'] > 1.5,
        fit_dict['b_prime_eq'] < 3.0,
        fit_dict['bulkmodul_eq'] > 174,
        fit_dict['bulkmodul_eq'] < 184,
        fit_dict['energy_eq'] > -453.9,
        fit_dict['energy_eq'] < -453.5,
        fit_dict['volume_eq'] > 1207,
        fit_dict['volume_eq'] < 1213,
    ]
    if not all(lst):
        print(fit_dict)
    return lst


class TestParallelTwoCores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        count_lst = [22, 22, 22, 21, 21]
        element_lst = ["Fe", "Ni", "Cr", "Co", "Cu"]
        potential = '2021--Deluigi-O-R--Fe-Ni-Cr-Co-Cu--LAMMPS--ipr1'
        resource_path = os.path.join(os.path.dirname(__file__), "static")

        # Generate SQS Structure
        structure_template = bulk("Al", cubic=True).repeat([3, 3, 3])
        mole_fractions = {
            el: c / len(structure_template) for el, c in zip(element_lst, count_lst)
        }
        structure = stk.build.sqs_structures(
            structure=structure_template,
            mole_fractions=mole_fractions,
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

    def test_example_elastic_constants_parallel_cores_two(self):
        structure_opt_lst = pyr.optimize_structure_parallel(
            structure_list=[self.structure.copy()],
            potential_dataframe_list=[self.df_pot_selected],
            cores=2
        )

        # Calculate Elastic Constants
        fit_dict = pyr.calculate_energy_volume_curve_parallel(
            structure_list=structure_opt_lst,
            potential_dataframe_list=[self.df_pot_selected],
            num_points=11,
            fit_type="polynomial",
            fit_order=3,
            vol_range=0.05,
            axes=["x", "y", "z"],
            strains=None,
            cores=2,
            minimization_activated=False,
        )[0]

        self.assertEqual(len(structure_opt_lst[0]), sum(self.count_lst))
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))

    def test_example_elastic_constants_with_minimization_parallel_cores_two(self):
        fit_dict = pyr.calculate_energy_volume_curve_parallel(
            structure_list=[self.structure.copy()],
            potential_dataframe_list=[self.df_pot_selected],
            num_points=11,
            fit_type="polynomial",
            fit_order=3,
            vol_range=0.05,
            axes=["x", "y", "z"],
            strains=None,
            cores=2,
            minimization_activated=True,
        )[0]
        self.assertTrue(all(validate_fitdict(fit_dict=fit_dict)))


if __name__ == '__main__':
    unittest.main()