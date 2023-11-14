====================================================================
atomistics - Interfaces for atomistic simulation codes and workflows
====================================================================

:Author:  Jan Janssen
:Contact: janssen@mpie.de

The :code:`atomistics` package consists of two primary components. On the one hand it provides interfaces to atomistic
simulation codes. In alphabetical order:

* `Abinit <https://www.abinit.org>`_ - Plane wave density functional theory
* `EMT <https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html>`_ - Effective medium theory potential
* `GPAW <https://wiki.fysik.dtu.dk/gpaw/>`_ - Density functional theory Python code based on the projector-augmented wave method
* `LAMMPS <https://www.lammps.org>`_ - Molecular Dynamics
* `Quantum Espresso <https://www.quantum-espresso.org>`_ - Integrated suite of Open-Source computer codes for electronic-structure calculations
* `Siesta <https://siesta-project.org>`_ - Electronic structure calculations and ab initio molecular dynamics

For majority of these simulation codes the :code:`atomistics` package use the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_
to interface the underlying C/ C++ and Fortran Codes with the Python programming language. Still this approach limits
the functionality of the simulation code to calculating the energy and forces, so by adding custom interfaces the
:code:`atomistics` package can support built-in features of the simulation code like structure optimization and molecular
dynamics.

On the other hand the :code:`atomistics` package also provides workflows to calculate material properties on the
atomistic scales, these include:

* equation of state - to calculate equilibrium properties like the equilibrium energy, equilibrium volume, equilibrium bulk modulus and its pressure derivative.
* elastic matrix - to calculate the elastic constants and elastic moduli.
* harmonic and quasi-harmonic approximation - to calculate the density of states, vibrational free energy and thermal expansion based on the finite displacements method implemented in `phonopy <https://phonopy.github.io/phonopy/>`_.
* molecular dynamics - to calculate finite temperature properties like thermal expansion including the anharmonic contributions.

All these workflows can be coupled with all the simulation codes implemented in the :code:`atomistics` package. In contrast
to the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ which provides similar functionality the focus
of the :code:`atomistics` package is not to reimplement existing functionality but rather simplify the process of coupling
existing simulation codes with existing workflows. Here the `phonopy <https://phonopy.github.io/phonopy/>`_ workflow is a
great example to enable the calculation of thermodynamic properties with the harmonic and quasi-harmonic approximation.

Example
-------
Calculate::

    from ase.build import bulk
    from atomistics.calculators import evaluate_with_ase
    from atomistics.workflows import EnergyVolumeCurveWorkflow
    from gpaw import GPAW, PW

    calculator = EnergyVolumeCurveWorkflow(
        structure=bulk("Al", a=4.05, cubic=True),
        num_points=11,
        fit_type='polynomial',
        fit_order=3,
        vol_range=0.05,
        axes=['x', 'y', 'z'],
        strains=None,
    )
    task_dict = calculator.generate_structures()
    result_dict = evaluate_with_ase(
        task_dict=task_dict,
        ase_calculator=GPAW(
            xc="PBE",
            mode=PW(300),
            kpts=(3, 3, 3)
        )
    )
    fit_dict = calculator.analyse_structures(output_dict=result_dict)

Documentation
-------------

.. toctree::
   :maxdepth: 2

   installation
   simulationcodes
   materialproperties

* :ref:`modindex`