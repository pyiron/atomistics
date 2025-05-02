import warnings


def raise_warning(module_list, import_error):
    if len(module_list) == 1:
        warnings.warn(
            message=module_list[0]
            + "() is not available as the import of the"
            + import_error.msg[2:]
            + " failed.",
            stacklevel=2,
        )
    else:
        error_msg = "(), ".join(module_list[:-1]) + "() and " + module_list[-1] + "()"
        warnings.warn(
            message=error_msg
            + " are not available as the import of the"
            + import_error.msg[2:]
            + " failed.",
            stacklevel=2,
        )
