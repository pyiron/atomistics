import numpy as np

from atomistics.workflows.evcurve.workflow import (
    EnergyVolumeCurveWorkflow,
    fit_ev_curve,
)
from atomistics.workflows.phonons.workflow import PhonopyWorkflow
from atomistics.workflows.phonons.units import (
    VaspToTHz,
    kJ_mol_to_eV,
    THzToEv,
    kb,
    EvTokJmol,
)


def get_free_energy(frequency, temperature):
    return kb * temperature * np.log(frequency / (kb * temperature))


class QuasiHarmonicWorkflow(EnergyVolumeCurveWorkflow):
    def __init__(
        self,
        structure,
        num_points=11,
        vol_range=0.05,
        fit_type="polynomial",
        fit_order=3,
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
            fit_type=fit_type,
            fit_order=fit_order,
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

    def get_thermal_properties(
        self,
        t_min=1,
        t_max=1500,
        t_step=50,
        temperatures=None,
        cutoff_frequency=None,
        pretend_real=False,
        band_indices=None,
        is_projection=False,
    ):
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
                t_step=t_step,
                t_max=t_max,
                t_min=t_min,
                temperatures=temperatures,
                cutoff_frequency=cutoff_frequency,
                pretend_real=pretend_real,
                band_indices=band_indices,
                is_projection=is_projection,
            )
        return tp_collect_dict

    def get_thermal_properties_classical(
        self,
        t_min=1,
        t_max=1500,
        t_step=50,
        temperatures=None,
        cutoff_frequency=None,
    ):
        if temperatures is None:
            temperatures = np.arange(t_min, t_max, t_step)
        if cutoff_frequency is None or cutoff_frequency < 0:
            cutoff_frequency = 0.0
        else:
            cutoff_frequency = cutoff_frequency * THzToEv
        tp_collect_dict = {}
        for strain, phono in self._phonopy_dict.items():
            t_property_lst = []
            for t in temperatures:
                t_property = 0.0
                for freqs, w in zip(
                    phono.phonopy.mesh.frequencies, phono.phonopy.mesh.weights
                ):
                    freqs = np.array(freqs) * THzToEv
                    cond = freqs > cutoff_frequency
                    t_property += (
                        np.sum(get_free_energy(frequency=freqs[cond], temperature=t))
                        * w
                    )
                t_property_lst.append(
                    t_property / np.sum(phono.phonopy.mesh.weights) * EvTokJmol
                )
            tp_collect_dict[strain] = {
                "temperatures": temperatures,
                "free_energy": np.array(t_property_lst) * kJ_mol_to_eV,
            }
        return tp_collect_dict

    def get_thermal_expansion(
        self,
        output_dict,
        t_min=1,
        t_max=1500,
        t_step=50,
        temperatures=None,
        cutoff_frequency=None,
        pretend_real=False,
        band_indices=None,
        is_projection=False,
        quantum_mechanical=True,
    ):
        (
            eng_internal_dict,
            mesh_collect_dict,
            dos_collect_dict,
        ) = self.analyse_structures(output_dict=output_dict)
        if quantum_mechanical:
            tp_collect_dict = self.get_thermal_properties(
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                temperatures=temperatures,
                cutoff_frequency=cutoff_frequency,
                pretend_real=pretend_real,
                band_indices=band_indices,
                is_projection=is_projection,
            )
        else:
            if is_projection:
                raise ValueError(
                    "is_projection!=False is incompatible to quantum_mechanical=False."
                )
            if pretend_real:
                raise ValueError(
                    "pretend_real!=False is incompatible to quantum_mechanical=False."
                )
            if band_indices is not None:
                raise ValueError(
                    "band_indices!=None is incompatible to quantum_mechanical=False."
                )

            tp_collect_dict = self.get_thermal_properties_classical(
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                temperatures=temperatures,
                cutoff_frequency=cutoff_frequency,
            )

        temperatures = tp_collect_dict[1.0]["temperatures"]
        strain_lst = eng_internal_dict.keys()
        volume_lst = self.get_volume_lst()
        eng_int_lst = np.array(list(eng_internal_dict.values()))

        vol_lst, eng_lst = [], []
        for i, temp in enumerate(temperatures):
            free_eng_lst = (
                np.array([tp_collect_dict[s]["free_energy"][i] for s in strain_lst])
                + eng_int_lst
            )
            fit_dict = fit_ev_curve(
                volume_lst=volume_lst,
                energy_lst=free_eng_lst,
                fit_type=self.fit_type,
                fit_order=self.fit_order,
            )
            eng_lst.append(fit_dict["energy_eq"])
            vol_lst.append(fit_dict["volume_eq"])
        return temperatures, vol_lst
