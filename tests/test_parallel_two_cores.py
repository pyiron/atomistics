import os
import unittest
import pyiron_lammps as pyr


def validate_elastic_constants(elastic_matrix):
    return [
        elastic_matrix[0, 0] > 200,
        elastic_matrix[1, 1] > 200,
        elastic_matrix[2, 2] > 200,
        elastic_matrix[0, 1] > 135,
        elastic_matrix[0, 2] > 135,
        elastic_matrix[1, 2] > 135,
        elastic_matrix[3, 3] > 100,
        elastic_matrix[4, 4] > 100,
        elastic_matrix[5, 5] > 100,
    ]


class TestParallelTwoCores(unittest.TestCase):
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

    def test_example_elastic_constants_parallel_cores_two(self):
        structure_opt_lst = pyr.optimize_structure_parallel(
            structure_list=[self.structure.copy()],
            potential_dataframe=self.df_pot_selected,
            cores=2
        )

        # Calculate Elastic Constants
        elastic_matrix = pyr.calculate_elastic_constants_parallel(
            structure_list=structure_opt_lst,
            potential_dataframe=self.df_pot_selected,
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
            fit_order=2,
            cores=2
        )[0]

        self.assertEqual(len(structure_opt_lst[0]), sum(self.count_lst))
        self.assertTrue(all(validate_elastic_constants(elastic_matrix=elastic_matrix)))

    def test_example_elastic_constants_with_minimization_parallel_cores_two(self):
        elastic_matrix = pyr.calculate_elastic_constants_with_minimization_parallel(
            structure_list=[self.structure.copy()],
            potential_dataframe=self.df_pot_selected,
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
            fit_order=2,
            cores=2
        )[0]
        self.assertTrue(all(validate_elastic_constants(elastic_matrix=elastic_matrix)))


if __name__ == '__main__':
    unittest.main()