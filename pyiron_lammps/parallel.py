from pympipool import Pool
from pyiron_lammps.wrapper import PyironLammpsLibrary
from pyiron_lammps.calculation import (
    optimize_structure,
    calculate_elastic_constants,
    calculate_elastic_constants_with_minimization,
)


def _get_lammps_mpi(enable_mpi=True):
    if enable_mpi:
        # To get the right instance of MPI.COMM_SELF it is necessary to import it inside the function.
        from mpi4py import MPI

        return PyironLammpsLibrary(
            working_directory=None,
            cores=1,
            comm=MPI.COMM_SELF,
            logger=None,
            log_file=None,
            library=None,
            diable_log_file=True,
        )
    else:
        return PyironLammpsLibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=None,
            log_file=None,
            library=None,
            diable_log_file=True,
        )


def _parallel_execution(function, input_parameter_lst, cores=1):
    if cores == 1:
        return [
            function(input_parameter=input_parameter + [False])
            for input_parameter in input_parameter_lst
        ]
    elif cores > 1:
        with Pool(cores=cores) as p:
            return p.map(
                function=function,
                lst=[
                    input_parameter + [True] for input_parameter in input_parameter_lst
                ],
            )
    else:
        raise ValueError("The number of cores has to be a positive integer.")


def _optimize_structure_serial(input_parameter):
    structure, potential_dataframe, enable_mpi = input_parameter
    return optimize_structure(
        lmp=_get_lammps_mpi(enable_mpi=enable_mpi),
        structure=structure,
        potential_dataframe=potential_dataframe,
    )


def _calculate_elastic_constants_serial(input_parameter):
    (
        structure,
        potential_dataframe,
        num_of_point,
        eps_range,
        sqrt_eta,
        fit_order,
        enable_mpi,
    ) = input_parameter
    return calculate_elastic_constants(
        lmp=_get_lammps_mpi(enable_mpi=enable_mpi),
        structure=structure,
        potential_dataframe=potential_dataframe,
        num_of_point=num_of_point,
        eps_range=eps_range,
        sqrt_eta=sqrt_eta,
        fit_order=fit_order,
    )


def _calculate_elastic_constants_with_minimization_serial(input_parameter):
    (
        structure,
        potential_dataframe,
        num_of_point,
        eps_range,
        sqrt_eta,
        fit_order,
        enable_mpi,
    ) = input_parameter
    return calculate_elastic_constants_with_minimization(
        lmp=_get_lammps_mpi(enable_mpi=enable_mpi),
        structure=structure,
        potential_dataframe=potential_dataframe,
        num_of_point=num_of_point,
        eps_range=eps_range,
        sqrt_eta=sqrt_eta,
        fit_order=fit_order,
    )


def optimize_structure_parallel(structure_list, potential_dataframe, cores=1):
    return _parallel_execution(
        function=_optimize_structure_serial,
        input_parameter_lst=[
            [structure, potential_dataframe] for structure in structure_list
        ],
        cores=cores,
    )


def calculate_elastic_constants_parallel(
    structure_list,
    potential_dataframe,
    num_of_point=5,
    eps_range=0.005,
    sqrt_eta=True,
    fit_order=2,
    cores=1,
):
    return _parallel_execution(
        function=_calculate_elastic_constants_serial,
        input_parameter_lst=[
            [
                structure,
                potential_dataframe,
                num_of_point,
                eps_range,
                sqrt_eta,
                fit_order,
            ]
            for structure in structure_list
        ],
        cores=cores,
    )


def calculate_elastic_constants_with_minimization_parallel(
    structure_list,
    potential_dataframe,
    num_of_point=5,
    eps_range=0.005,
    sqrt_eta=True,
    fit_order=2,
    cores=1,
):
    return _parallel_execution(
        function=_calculate_elastic_constants_with_minimization_serial,
        input_parameter_lst=[
            [
                structure,
                potential_dataframe,
                num_of_point,
                eps_range,
                sqrt_eta,
                fit_order,
            ]
            for structure in structure_list
        ],
        cores=cores,
    )
