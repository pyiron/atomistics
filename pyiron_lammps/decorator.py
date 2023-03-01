from pyiron_lammps.wrapper import PyironLammpsLibrary


def calculation(funct):
    def funct_return(lmp=None, *args, **kwargs):
        # Create temporary LAMMPS instance if necessary
        if lmp is None:
            close_lmp_after_calculation = True
            lmp = PyironLammpsLibrary()
        else:
            close_lmp_after_calculation = False

        # Run function
        result = funct(lmp=lmp, *args, **kwargs)

        # Close temporary LAMMPS instance
        if close_lmp_after_calculation:
            lmp.close()
        return result

    return funct_return
