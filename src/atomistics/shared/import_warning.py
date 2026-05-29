import warnings


def raise_warning(module_list, import_error):
    """
    Issue a UserWarning when one or more optional module functions are unavailable.

    Args:
        module_list (list[str]): Names of the functions that could not be imported.
        import_error (ImportError): The import error that caused the unavailability.
    """
    if len(module_list) == 1:
        warnings.warn(
            message=module_list[0]
            + "() is not available as the following import failed: "
            + str(import_error),
            stacklevel=2,
        )
    else:
        error_msg = "(), ".join(module_list[:-1]) + "() and " + module_list[-1] + "()"
        warnings.warn(
            message=error_msg
            + " are not available as the following import failed: "
            + str(import_error),
            stacklevel=2,
        )
