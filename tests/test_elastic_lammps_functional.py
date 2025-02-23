import os

from ase.build import bulk
import numpy as np
import unittest

from atomistics.workflows.elastic.workflow import (
    analyse_structures_helper,
    generate_structures_helper,
)

try:
    from atomistics.calculators import evaluate_with_lammpslib, get_potential_by_name

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(
    skip_lammps_test, "LAMMPS is not installed, so the LAMMPS tests are skipped."
)
class TestElastic(unittest.TestCase):
    def test_calc_elastic_functions(self):
        structure = bulk("Al", cubic=True)
        df_pot_selected = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=os.path.join(os.path.dirname(__file__), "static", "lammps"),
        )
        result_dict = evaluate_with_lammpslib(
            task_dict={"optimize_positions_and_volume": structure},
            potential_dataframe=df_pot_selected,
        )
        sym_dict, structure_dict = generate_structures_helper(
            structure=result_dict["structure_with_optimized_positions_and_volume"],
            eps_range=0.005,
            num_of_point=5,
            zero_strain_job_name="s_e_0",
            sqrt_eta=True,
        )
        result_dict = evaluate_with_lammpslib(
            task_dict={"calc_energy": structure_dict},
            potential_dataframe=df_pot_selected,
        )
        sym_dict, elastic_dict = analyse_structures_helper(
            output_dict=result_dict,
            sym_dict=sym_dict,
            fit_order=2,
            zero_strain_job_name="s_e_0",
        )
        self.assertTrue(
            np.isclose(
                elastic_dict["elastic_matrix"],
                np.array(
                    [
                        [114.10311701, 60.51102935, 60.51102935, 0.0, 0.0, 0.0],
                        [60.51102935, 114.10311701, 60.51102935, 0.0, 0.0, 0.0],
                        [60.51102935, 60.51102935, 114.10311701, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 31.67489592, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 31.67489592, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 31.67489592],
                    ]
                ),
            ).all()
        )
        self.assertEqual(sym_dict["SGN"], 225)
        self.assertEqual(sym_dict["LC"], "CI")
        self.assertEqual(sym_dict["Lag_strain_list"], ["01", "08", "23"])
        self.assertTrue(
            np.isclose(
                sym_dict["epss"], np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
            ).all()
        )
        self.assertAlmostEqual(sym_dict["v0"], 66.43035441556098)
        self.assertAlmostEqual(sym_dict["e0"], -13.439999952735112)
        self.assertTrue(
            np.isclose(
                sym_dict["strain_energy"],
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
                            (0.005, -13.438204192615178),
                        ],
                        [
                            (-0.005, -13.437971451918395),
                            (-0.0025, -13.439501038418321),
                            (0.0, -13.439999952735112),
                            (0.0024999999999999996, -13.439515785430654),
                            (0.005, -13.438089441277942),
                        ],
                    ]
                ),
            ).all()
        )
        self.assertTrue(
            np.isclose(
                sym_dict["A2"], np.array([[2.20130388, 1.08985578, 1.1861949]])
            ).all()
        )
        self.assertAlmostEqual(elastic_dict["bulkmodul_voigt"], 78.37505857279467)
        self.assertAlmostEqual(elastic_dict["shearmodul_voigt"], 29.723355082523486)
        self.assertAlmostEqual(elastic_dict["youngsmodul_voigt"], 79.16270932956571)
        self.assertAlmostEqual(elastic_dict["poissonsratio_voigt"], 0.3316583728482119)
        self.assertTrue(
            np.isclose(
                elastic_dict["elastic_matrix_inverse"],
                np.array(
                    [
                        [0.01385733, -0.00480214, -0.00480214, 0.0, 0.0, 0.0],
                        [-0.00480214, 0.01385733, -0.00480214, 0.0, 0.0, 0.0],
                        [-0.00480214, -0.00480214, 0.01385733, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.03157074, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.03157074, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.03157074],
                    ]
                ),
            ).all()
        )
        self.assertAlmostEqual(elastic_dict["bulkmodul_reuss"], 78.37505857279469)
        self.assertAlmostEqual(elastic_dict["shearmodul_reuss"], 29.524633432574454)
        self.assertAlmostEqual(elastic_dict["youngsmodul_reuss"], 78.69249533327846)
        self.assertAlmostEqual(elastic_dict["poissonsratio_reuss"], 0.3326582955380103)
        self.assertAlmostEqual(elastic_dict["bulkmodul_hill"], 78.37505857279467)
        self.assertAlmostEqual(elastic_dict["shearmodul_hill"], 29.623994257548972)
        self.assertAlmostEqual(elastic_dict["youngsmodul_hill"], 78.9276905674883)
        self.assertAlmostEqual(elastic_dict["poissonsratio_hill"], 0.33215814655674686)
        self.assertAlmostEqual(elastic_dict["AVR"], 0.3354065765429198)
