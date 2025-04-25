from unittest import TestCase
from atomistics.shared.import_warning import raise_warning


class TestImportWarning(TestCase):
    def test_import_warning(self):
        with self.assertWarns(Warning):
            try:
                import this_package_does_not_exist
            except ImportError as e:
                raise_warning(module_list=["a"], import_error=e)
        with self.assertWarns(Warning):
            try:
                import this_package_does_not_exist
            except ImportError as e:
                raise_warning(module_list=["a", "b", "c"], import_error=e)
