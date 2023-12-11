import dataclasses

from pylammpsmpi import LammpsASELibrary


@dataclasses.dataclass
class LammpsOutput:
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    def __call__(self, engine: LammpsASELibrary, quantity: str):
        return getattr(self, quantity)(engine)


@dataclasses.dataclass
class LammpsMDQuantityGetter(LammpsOutput):
    positions: callable = LammpsASELibrary.interactive_positions_getter
    cell: callable = LammpsASELibrary.interactive_cells_getter
    forces: callable = LammpsASELibrary.interactive_forces_getter
    temperature: callable = LammpsASELibrary.interactive_temperatures_getter
    energy_pot: callable = LammpsASELibrary.interactive_energy_pot_getter
    energy_tot: callable = LammpsASELibrary.interactive_energy_tot_getter
    pressure: callable = LammpsASELibrary.interactive_pressures_getter
    velocities: callable = LammpsASELibrary.interactive_velocities_getter


@dataclasses.dataclass
class LammpsStaticQuantityGetter(LammpsOutput):
    forces: callable = LammpsASELibrary.interactive_forces_getter
    energy: callable = LammpsASELibrary.interactive_energy_pot_getter
    stress: callable = LammpsASELibrary.interactive_pressures_getter


quantity_getter_md = LammpsMDQuantityGetter()
quantities_md = quantity_getter_md.fields()
quantity_getter_static = LammpsStaticQuantityGetter()
quantities_static = quantity_getter_static.fields()


def get_quantity(lmp_instance, quantity_getter, quantities):
    return {q: quantity_getter(lmp_instance, q) for q in quantities}


def get_static_output(lmp_instance, quantities=quantities_static):
    return get_quantity(
        lmp_instance=lmp_instance,
        quantity_getter=quantity_getter_static,
        quantities=quantities,
    )


def get_md_output(lmp_instance, quantities=quantities_md):
    return get_quantity(
        lmp_instance=lmp_instance,
        quantity_getter=quantity_getter_md,
        quantities=quantities,
    )
