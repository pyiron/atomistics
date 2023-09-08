import os

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows.elastic.workflow import ElasticMatrixWorkflow

try:
    from atomistics.calculators.lammps_library.calculator import evaluate_with_lammps
    from atomistics.calculators.lammps_library.potential import get_potential_dataframe

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        potential = '1999--Mishin-Y--Al--LAMMPS--ipr1'
        resource_path = os.path.join(os.path.dirname(__file__), "static", "lammps")
        structure = bulk("Al", a=4.05, cubic=True)
        df_pot = get_potential_dataframe(
            structure=structure,
            resource_path=resource_path
        )
        df_pot_selected = df_pot[df_pot.Name == potential].iloc[0]
        calculator = ElasticMatrixWorkflow(
            structure=structure,
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
            fit_order=2
        )
        structure_dict = calculator.generate_structures()
        result_dict = evaluate_with_lammps(
            task_dict=structure_dict,
            potential_dataframe=df_pot_selected,
        )
        elastic_dict = calculator.analyse_structures(output_dict=result_dict)
        self.assertTrue(np.isclose(elastic_dict["C"][0, 0], 114.10393023))
        self.assertTrue(np.isclose(elastic_dict["C"][0, 1], 60.51098897))
        self.assertTrue(np.isclose(elastic_dict["C"][3, 3], 51.23931149))
