import unittest

from atomistics.workflows.interface import Workflow


class WorkflowClass(Workflow):
    def generate_structures(self) -> dict:
        raise NotImplementedError()

    def analyse_structures(self, output_dict: dict) -> None:
        raise NotImplementedError()


class TestWorkflowClass(unittest.TestCase):
    def test_generate_structures(self):
        with self.assertRaises(NotImplementedError):
            temp = WorkflowClass()
            temp.generate_structures()

    def test_analyse_structures(self):
        with self.assertRaises(NotImplementedError):
            temp = WorkflowClass()
            temp.analyse_structures(output_dict={})
