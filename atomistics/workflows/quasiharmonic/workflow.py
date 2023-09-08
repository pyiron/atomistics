from atomistics.workflows.evcurve.workflow import EnergyVolumeCurveWorkflow
from atomistics.workflows.phonons.workflow import PhonopyWorkflow
from atomistics.workflows.phonons.units import VaspToTHz


class QuasiHarmonicWorkflow(EnergyVolumeCurveWorkflow):
    def __init__(
        self,
        structure,
        num_points=11,
        vol_range=0.05,
        interaction_range=10,
        factor=VaspToTHz,
        displacement=0.01,
        dos_mesh=20,
        primitive_matrix=None,
        number_of_snapshots=None,
    ):
        super().__init__(
            structure=structure,
            num_points=num_points,
            fit_type="polynomial",
            fit_order=3,
            vol_range=vol_range,
            axes=["x", "y", "z"],
            strains=None,
        )
        self._interaction_range = interaction_range
        self._displacement = displacement
        self._dos_mesh = dos_mesh
        self._number_of_snapshots = number_of_snapshots
        self._factor = factor
        self._primitive_matrix = primitive_matrix
        self._phonopy_dict = {}

    def generate_structures(self):
        task_dict = super().generate_structures()
        task_dict["calc_forces"] = {}
        for strain, structure in task_dict["calc_energy"].items():
            self._phonopy_dict[strain] = PhonopyWorkflow(
                structure=structure,
                interaction_range=self._interaction_range,
                factor=self._factor,
                displacement=self._displacement,
                dos_mesh=self._dos_mesh,
                primitive_matrix=self._primitive_matrix,
                number_of_snapshots=self._number_of_snapshots,
            )
            structure_task_dict = self._phonopy_dict[strain].generate_structures()
            task_dict["calc_forces"].update(
                {
                    (strain, key): structure_phono
                    for key, structure_phono in structure_task_dict[
                        "calc_forces"
                    ].items()
                }
            )
        return task_dict

    def analyse_structures(self, output_dict):
        eng_internal_dict = output_dict["energy"]
        mesh_collect_dict, dos_collect_dict = {}, {}
        for strain, phono in self._phonopy_dict.items():
            mesh_dict, dos_dict = phono.analyse_structures(
                output_dict={
                    k: v for k, v in output_dict["forces"].items() if strain in k
                }
            )
            mesh_collect_dict[strain] = mesh_dict
            dos_collect_dict[strain] = dos_dict
        return eng_internal_dict, mesh_collect_dict, dos_collect_dict

    def get_thermal_properties(self, t_min=1, t_max=1500, t_step=50, temperatures=None):
        """
        Returns thermal properties at constant volume in the given temperature range.  Can only be called after job
        successfully ran.

        Args:
            t_min (float): minimum sample temperature
            t_max (float): maximum sample temperature
            t_step (int):  temperature sample interval
            temperatures (array_like, float):  custom array of temperature samples, if given t_min, t_max, t_step are
                                               ignored.

        Returns:
            :class:`Thermal`: thermal properties as returned by Phonopy
        """
        tp_collect_dict = {}
        for strain, phono in self._phonopy_dict.items():
            tp_collect_dict[strain] = phono.get_thermal_properties(
                t_step=t_step, t_max=t_max, t_min=t_min, temperatures=temperatures
            )
        return tp_collect_dict
