import unittest

from ase.build import bulk

from atomistics.workflows.elastic.helper import get_tasks_for_elastic_matrix


class TestElasticWorkflowHelper(unittest.TestCase):
    def setUp(self):
        self.structure = bulk("Al", a=4.05, cubic=True)

    def test_too_large_deformation_raises(self):
        with self.assertRaises(Exception):
            get_tasks_for_elastic_matrix(
                structure=self.structure, eps_range=5.0, num_of_point=5
            )


if __name__ == "__main__":
    unittest.main()
