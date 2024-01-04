====================================================================
atomistics - Interfaces for atomistic simulation codes and workflows
====================================================================

:Author:  Jan Janssen
:Contact: janssen@mpie.de

The :code:`atomistics` package consists of two primary components. On the one hand it provides interfaces to atomistic
simulation codes - named :code:`calculators`. The supported simulation codes in alphabetical order are:

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

On the other hand the :code:`atomistics` package also provides :code:`workflows` to calculate material properties on the
atomistic scales, these include:

* `Equation of State <https://atomistics.readthedocs.io/en/latest/workflows.html#energy-volume-curve>`_ - to calculate equilibrium properties like the equilibrium energy, equilibrium volume, equilibrium bulk modulus and its pressure derivative.
* `Elastic Matrix <https://atomistics.readthedocs.io/en/latest/workflows.html#elastic-matrix>`_ - to calculate the elastic constants and elastic moduli.
* `Harmonic and Quasi-harmonic Approximation <https://atomistics.readthedocs.io/en/latest/workflows.html#harmonic-approximation>`_ - to calculate the density of states, vibrational free energy and thermal expansion based on the finite displacements method implemented in `phonopy <https://phonopy.github.io/phonopy/>`_.
* `Molecular Dynamics <https://atomistics.readthedocs.io/en/latest/workflows.html#molecular-dynamics>`_ - to calculate finite temperature properties like thermal expansion including the anharmonic contributions.

All these :code:`workflows` can be coupled with all the simulation codes implemented in the :code:`atomistics` package.
In contrast to the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ which provides similar functionality
the focus of the :code:`atomistics` package is not to reimplement existing functionality but rather simplify the process
of coupling existing simulation codes with existing workflows. Here the `phonopy <https://phonopy.github.io/phonopy/>`_
workflow is a great example to enable the calculation of thermodynamic properties with the harmonic and quasi-harmonic
approximation.

Example
-------
Use the Equation of State to calculate the equilibrium properties like the equilibrium volume, equilibrium energy,
equilibrium bulk modulus and its derivative using the `GPAW <https://wiki.fysik.dtu.dk/gpaw/>`_ simulation code::

    from ase.build import bulk
    from atomistics.calculators import evaluate_with_ase
    from atomistics.workflows import EnergyVolumeCurveWorkflow
    from gpaw import GPAW, PW

    workflow = EnergyVolumeCurveWorkflow(
        structure=bulk("Al", a=4.05, cubic=True),
        num_points=11,
        fit_type='polynomial',
        fit_order=3,
        vol_range=0.05,
        axes=['x', 'y', 'z'],
        strains=None,
    )
    task_dict = workflow.generate_structures()
    print(task_dict)
    >>> {'calc_energy': OrderedDict([
    >>>     (0.95, Atoms(symbols='Al4', pbc=True, cell=[3.9813426685908118, 3.9813426685908118, 3.9813426685908118])),
    >>>     (0.96, Atoms(symbols='Al4', pbc=True, cell=[3.9952635604153612, 3.9952635604153612, 3.9952635604153612])),
    >>>     (0.97, Atoms(symbols='Al4', pbc=True, cell=[4.009088111958974, 4.009088111958974, 4.009088111958974])),
    >>>     (0.98, Atoms(symbols='Al4', pbc=True, cell=[4.022817972936038, 4.022817972936038, 4.022817972936038])),
    >>>     (0.99, Atoms(symbols='Al4', pbc=True, cell=[4.036454748321015, 4.036454748321015, 4.036454748321015])),
    >>>     (1.0, Atoms(symbols='Al4', pbc=True, cell=[4.05, 4.05, 4.05])),
    >>>     (1.01, Atoms(symbols='Al4', pbc=True, cell=[4.063455248345461, 4.063455248345461, 4.063455248345461])),
    >>>     (1.02, Atoms(symbols='Al4', pbc=True, cell=[4.076821973718458, 4.076821973718458, 4.076821973718458])),
    >>>     (1.03, Atoms(symbols='Al4', pbc=True, cell=[4.0901016179023415, 4.0901016179023415, 4.0901016179023415])),
    >>>     (1.04, Atoms(symbols='Al4', pbc=True, cell=[4.1032955854717175, 4.1032955854717175, 4.1032955854717175])),
    >>>     (1.05, Atoms(symbols='Al4', pbc=True, cell=[4.1164052451001565, 4.1164052451001565, 4.1164052451001565]))
    >>> ])}

In the first step the :code:`EnergyVolumeCurveWorkflow` object is initialized including all the parameters to generate
the strained structures and afterwards fit the resulting energy volume curve. This allows the user to see all relevant
parameters at one place. After the initialization the function :code:`generate_structures()` is called without any
additional parameters. This function returns the task dictionary :code:`task_dict` which includes the tasks which should
be executed by the calculator. In this case the task is to calculate the energy :code:`calc_energy` of the eleven
generated structures. Each structure is labeled by the ratio of compression or elongation. In the second step the
:code:`task_dict` is evaluate with the `GPAW <https://wiki.fysik.dtu.dk/gpaw/>`_ simulation code using the
:code:`evaluate_with_ase()` function::

    result_dict = evaluate_with_ase(
        task_dict=task_dict,
        ase_calculator=GPAW(
            xc="PBE",
            mode=PW(300),
            kpts=(3, 3, 3)
        )
    )
    print(result_dict)
    >>> {'energy': {
    >>>     0.95: -14.895378072824752,
    >>>     0.96: -14.910819737657118,
    >>>     0.97: -14.922307241122466,
    >>>     0.98: -14.930392279321056,
    >>>     0.99: -14.935048569964911,
    >>>     1.0: -14.936666396364169,
    >>>     1.01: -14.935212782128556,
    >>>     1.02: -14.931045138839849,
    >>>     1.03: -14.924165445706581,
    >>>     1.04: -14.914703574005678,
    >>>     1.05: -14.902774559134226
    >>> }}

In analogy to the :code:`task_dict` which defines the tasks to be executed by the simulation code the :code:`result_dict`
summarizes the results of the calculations. In this case the energies calculated for the specific strains. By ordering
both the :code:`task_dict` and the :code:`result_dict` with the same labels, the :code:`EnergyVolumeCurveWorkflow` object
is able to match the calculation results to the corresponding structure. Finally, in the third step the :code:`analyse_structures()`
function takes the :code:`result_dict` as an input and fits the Equation of State with the fitting parameters defined in
the first step::

    fit_dict = workflow.analyse_structures(output_dict=result_dict)
    print(fit_dict)
    >>> {'poly_fit': array([-9.30297838e-05,  2.19434659e-02, -1.68388816e+00,  2.73605421e+01]),
    >>>  'fit_type': 'polynomial',
    >>>  'fit_order': 3,
    >>>  'volume_eq': 66.44252286131888,
    >>>  'energy_eq': -14.93670322204575,
    >>>  'bulkmodul_eq': 72.38919826304497,
    >>>  'b_prime_eq': 4.45383655040775,
    >>>  'least_square_error': 4.432974529908853e-09,
    >>>  'volume': [63.10861874999998, 63.77291999999998, ..., 69.75163125000002],
    >>>  'energy': [-14.895378072824752, -14.910819737657118, ..., -14.902774559134226]
    >>> }

As a result the equilibrium parameters are returned plus the parameters of the polynomial and the set of volumes and
energies which were fitted to achieve these results. The important step here is that while the interface between the
first and the second as well as between the second and the third step is clearly defined independent of the specific
workflow, the initial parameters for the workflow to initialize the :code:`EnergyVolumeCurveWorkflow` object as well as
the final output of the :code:`fit_dict` are workflow specific.

Disclaimer
----------
While we try to develop a stable and reliable software library, the development remains a opensource project under the
BSD 3-Clause License without any warranties::

    BSD 3-Clause License

    Copyright (c) 2023, Jan Janssen
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Documentation
-------------

.. toctree::
   :maxdepth: 2

   installation
   simulationcodes
   workflows
   materialproperties

* :ref:`modindex`