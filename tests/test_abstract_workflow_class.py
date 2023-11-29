import unittest

from atomistics.workflows.shared.workflow import Workflow


class WorkflowClass(Workflow):
    pass


class TestWorkflowClass(unittest.TestCase):
    def test_generate_structures(self):
        with self.assertRaises(NotImplementedError):
            temp = WorkflowClass()
            temp.generate_structures()

    def test_analyse_structures(self):
        with self.assertRaises(NotImplementedError):
            temp = WorkflowClass()
            temp.analyse_structures(output_dict={})
