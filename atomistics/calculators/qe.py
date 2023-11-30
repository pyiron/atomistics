import os
import subprocess

from ase.io import write

try:
    from pwtools import io
except ImportError:
    pass

from atomistics.calculators.wrapper import as_task_dict_evaluator


def call_qe_via_ase_command(calculation_name, working_directory):
    qe_command = os.environ["ASE_ESPRESSO_COMMAND"].replace("PREFIX", calculation_name)
    subprocess.check_output(
        qe_command, shell=True, universal_newlines=True, cwd=working_directory
    )


def set_pseudo_potentials(pseudopotentials):
    if pseudopotentials is not None:
        return pseudopotentials
    else:
        raise ValueError()


def generate_input_data(**kwargs):
    return kwargs


def optimize_positions_and_volume_with_qe(
    structure,
    calculation_name="espresso",
    working_directory=".",
    kpts=(3, 3, 3),
    ecutwfc=29.49,
    conv_thr=1e-06,
    diagonalization="david",
    electron_maxstep=100,
    nstep=200,
    etot_conv_thr=1e-4,
    forc_conv_thr=1e-3,
    smearing="methfessel-paxton",
    pseudopotentials=None,
    tstress=True,
    tprnfor=True,
    **kwargs,
):
    input_data = generate_input_data(
        calculation="vc-relax",
        ecutwfc=ecutwfc,
        conv_thr=conv_thr,
        diagonalization=diagonalization,
        electron_maxstep=electron_maxstep,
        nstep=nstep,
        etot_conv_thr=etot_conv_thr,
        forc_conv_thr=forc_conv_thr,
        smearing=smearing,
        **kwargs,
    )
    pseudopotentials = set_pseudo_potentials(pseudopotentials=pseudopotentials)
    write(
        calculation_name + ".pwi",
        structure,
        Crystal=True,
        kpts=kpts,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        tstress=tstress,
        tprnfor=tprnfor,
    )
    call_qe_via_ase_command(
        calculation_name=calculation_name, working_directory=working_directory
    )
    return io.read_pw_md(calculation_name + ".pwo")[-1].get_ase_atoms()


def calc_energy_with_qe(
    structure,
    calculation_name="espresso",
    working_directory=".",
    kpts=(3, 3, 3),
    ecutwfc=29.49,
    conv_thr=1e-06,
    diagonalization="david",
    electron_maxstep=100,
    smearing="methfessel-paxton",
    pseudopotentials=None,
    tstress=True,
    tprnfor=True,
    **kwargs,
):
    input_data = generate_input_data(
        calculation="scf",
        ecutwfc=ecutwfc,
        conv_thr=conv_thr,
        diagonalization=diagonalization,
        electron_maxstep=electron_maxstep,
        smearing=smearing,
        **kwargs,
    )
    pseudopotentials = set_pseudo_potentials(pseudopotentials=pseudopotentials)
    write(
        calculation_name + ".pwi",
        structure,
        Crystal=True,
        kpts=kpts,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        tstress=tstress,
        tprnfor=tprnfor,
    )
    call_qe_via_ase_command(
        calculation_name=calculation_name, working_directory=working_directory
    )
    return io.read_pw_scf(calculation_name + ".pwo").etot


def calc_energy_and_forces_with_qe(
    structure,
    calculation_name="espresso",
    working_directory=".",
    kpts=(3, 3, 3),
    ecutwfc=29.49,
    conv_thr=1e-06,
    diagonalization="david",
    electron_maxstep=100,
    smearing="methfessel-paxton",
    pseudopotentials=None,
    tstress=True,
    tprnfor=True,
    **kwargs,
):
    input_data = generate_input_data(
        calculation="scf",
        ecutwfc=ecutwfc,
        conv_thr=conv_thr,
        diagonalization=diagonalization,
        electron_maxstep=electron_maxstep,
        smearing=smearing,
        **kwargs,
    )
    pseudopotentials = set_pseudo_potentials(pseudopotentials=pseudopotentials)
    write(
        calculation_name + ".pwi",
        structure,
        Crystal=True,
        kpts=kpts,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        tstress=tstress,
        tprnfor=tprnfor,
    )
    call_qe_via_ase_command(
        calculation_name=calculation_name, working_directory=working_directory
    )
    output = io.read_pw_scf(calculation_name + ".pwo")
    return output.etot, output.forces


def calc_forces_with_qe(
    structure,
    calculation_name="espresso",
    working_directory=".",
    kpts=(3, 3, 3),
    ecutwfc=29.49,
    conv_thr=1e-06,
    diagonalization="david",
    electron_maxstep=100,
    smearing="methfessel-paxton",
    pseudopotentials=None,
    tstress=True,
    tprnfor=True,
    **kwargs,
):
    input_data = generate_input_data(
        calculation="scf",
        ecutwfc=ecutwfc,
        conv_thr=conv_thr,
        diagonalization=diagonalization,
        electron_maxstep=electron_maxstep,
        smearing=smearing,
        **kwargs,
    )
    pseudopotentials = set_pseudo_potentials(pseudopotentials=pseudopotentials)
    write(
        calculation_name + ".pwi",
        structure,
        Crystal=True,
        kpts=kpts,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        tstress=tstress,
        tprnfor=tprnfor,
    )
    call_qe_via_ase_command(
        calculation_name=calculation_name, working_directory=working_directory
    )
    return io.read_pw_scf(calculation_name + ".pwo").forces


@as_task_dict_evaluator
def evaluate_with_qe(
    structure,
    tasks,
    calculation_name="espresso",
    working_directory=".",
    kpts=(3, 3, 3),
    ecutwfc=29.49,
    conv_thr=1e-06,
    diagonalization="david",
    electron_maxstep=100,
    smearing="methfessel-paxton",
    pseudopotentials=None,
    tstress=True,
    tprnfor=True,
    **kwargs,
):
    results = {}
    if "optimize_positions_and_volume" in tasks:
        results[
            "structure_with_optimized_positions_and_volume"
        ] = optimize_positions_and_volume_with_qe(
            structure=structure,
            calculation_name=calculation_name,
            working_directory=working_directory,
            kpts=kpts,
            ecutwfc=ecutwfc,
            conv_thr=conv_thr,
            diagonalization=diagonalization,
            electron_maxstep=electron_maxstep,
            smearing=smearing,
            pseudopotentials=pseudopotentials,
            tstress=tstress,
            tprnfor=tprnfor,
            **kwargs,
        )
    elif "calc_energy" in tasks or "calc_forces" in tasks:
        if "calc_energy" in tasks and "calc_forces" in tasks:
            results["energy"], results["forces"] = calc_energy_and_forces_with_qe(
                structure=structure,
                calculation_name=calculation_name,
                working_directory=working_directory,
                kpts=kpts,
                ecutwfc=ecutwfc,
                conv_thr=conv_thr,
                diagonalization=diagonalization,
                electron_maxstep=electron_maxstep,
                smearing=smearing,
                pseudopotentials=pseudopotentials,
                tstress=tstress,
                tprnfor=tprnfor,
                **kwargs,
            )
        elif "calc_energy" in tasks:
            results["energy"] = calc_energy_with_qe(
                structure=structure,
                calculation_name=calculation_name,
                working_directory=working_directory,
                kpts=kpts,
                ecutwfc=ecutwfc,
                conv_thr=conv_thr,
                diagonalization=diagonalization,
                electron_maxstep=electron_maxstep,
                smearing=smearing,
                pseudopotentials=pseudopotentials,
                tstress=tstress,
                tprnfor=tprnfor,
                **kwargs,
            )
        elif "calc_forces" in tasks:
            results["forces"] = calc_forces_with_qe(
                structure=structure,
                calculation_name=calculation_name,
                working_directory=working_directory,
                kpts=kpts,
                ecutwfc=ecutwfc,
                conv_thr=conv_thr,
                diagonalization=diagonalization,
                electron_maxstep=electron_maxstep,
                smearing=smearing,
                pseudopotentials=pseudopotentials,
                tstress=tstress,
                tprnfor=tprnfor,
                **kwargs,
            )
    else:
        raise ValueError("The Quantum Espresso calculator does not implement:", tasks)
    return results
