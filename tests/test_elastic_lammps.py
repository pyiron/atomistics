import os

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows import ElasticMatrixWorkflow, optimize_positions_and_volume

try:
    from atomistics.calculators import evaluate_with_lammps, get_potential_by_name

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestElastic(unittest.TestCase):
    def test_calc_elastic(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        task_dict = optimize_positions_and_volume(structure=structure)
        result_dict = evaluate_with_lammps(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        workflow = ElasticMatrixWorkflow(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            num_of_point=5,
            eps_range=0.005,
            sqrt_eta=True,
            fit_order=2,
        )
        task_dict = workflow.generate_structures()
        result_dict = evaluate_with_lammps(
            task_dict=task_dict,
            potential_dataframe=df_pot_selected,
        )
        elastic_dict = workflow.analyse_structures(output_dict=result_dict)
        self.assertTrue(
            np.isclose(
                elastic_dict["elastic_matrix"],
                np.array(
                    [
                        [114.10311701, 60.51102935, 60.51102935, 0.0, 0.0, 0.0],
                        [60.51102935, 114.10311701, 60.51102935, 0.0, 0.0, 0.0],
                        [60.51102935, 60.51102935, 114.10311701, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 51.23853765, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 51.23853765, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 51.23853765],
                    ]
                ),
            ).all()
        )
        self.assertEqual(workflow._data["SGN"], 225)
        self.assertEqual(workflow._data["LC"], "CI")
        self.assertEqual(workflow._data["Lag_strain_list"], ["01", "08", "23"])
        self.assertTrue(
            np.isclose(
                workflow._data["epss"], np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
            ).all()
        )
        self.assertAlmostEqual(workflow._data["v0"], 66.43035441556098)
        self.assertAlmostEqual(workflow._data["e0"], -13.439999952735112)
        self.assertTrue(
            np.isclose(
                workflow._data["strain_energy"],
                np.array(
                    [
                        [
                            (-0.005, -13.436320248980278),
                            (-0.0025, -13.439079680886989),
                            (0.0, -13.439999952735112),
                            (0.0024999999999999996, -13.439084974614394),
                            (0.005, -13.436364320399795),
                        ],
                        [
                            (-0.005, -13.43817471490433),
                            (-0.0025, -13.439544638502628),
                            (0.0, -13.439999952735112),
                            (0.0024999999999999996, -13.43954822781134),
                            (0.005, -13.438204192615181),
                        ],
                        [
                            (-0.005, -13.436741954502294),
                            (-0.0025, -13.439195465714551),
                            (0.0, -13.439999952735112),
                            (0.0024999999999999996, -13.439213491269701),
                            (0.005, -13.436885713447486),
                        ],
                    ]
                ),
            ).all()
        )
        self.assertTrue(
            np.isclose(
                workflow._data["A2"], np.array([2.20130388, 1.08985578, 1.91883479])
            ).all()
        )
        self.assertAlmostEqual(elastic_dict["bulkmodul_voigt"], 78.37505857279467)
        self.assertAlmostEqual(elastic_dict["shearmodul_voigt"], 41.46154012284969)
        self.assertAlmostEqual(elastic_dict["youngsmodul_voigt"], 105.73882997912072)
        self.assertAlmostEqual(elastic_dict["poissonsratio_voigt"], 0.2751435386362729)
        self.assertTrue(
            np.isclose(
                elastic_dict["elastic_matrix_inverse"],
                np.array(
                    [
                        [0.01385733, -0.00480214, -0.00480214, 0.0, 0.0, 0.0],
                        [-0.00480214, 0.01385733, -0.00480214, 0.0, 0.0, 0.0],
                        [-0.00480214, -0.00480214, 0.01385733, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.01951656, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.01951656, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.01951656],
                    ]
                ),
            ).all()
        )
        self.assertAlmostEqual(elastic_dict["bulkmodul_reuss"], 78.37505857279469)
        self.assertAlmostEqual(elastic_dict["shearmodul_reuss"], 37.54104251720356)
        self.assertAlmostEqual(elastic_dict["youngsmodul_reuss"], 97.11702764970639)
        self.assertAlmostEqual(elastic_dict["poissonsratio_reuss"], 0.29347803281170937)
        self.assertAlmostEqual(elastic_dict["bulkmodul_hill"], 78.37505857279467)
        self.assertAlmostEqual(elastic_dict["shearmodul_hill"], 39.501291320026624)
        self.assertAlmostEqual(elastic_dict["youngsmodul_hill"], 101.45869947879392)
        self.assertAlmostEqual(elastic_dict["poissonsratio_hill"], 0.2842453510798992)
        self.assertAlmostEqual(elastic_dict["AVR"], 4.962492964955925)
