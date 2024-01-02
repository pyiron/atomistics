from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses


class Output(ABC):
    """
    A base-class for holding output.

    Promises that each piece of data has its name stored in the :meth:`fields` class
    method, and that a subset of data can be easily accessed via the :meth:`get` method.
    """
    @classmethod
    @abstractmethod
    def fields(cls) -> tuple[str]:
        """
        The names of all the data held by this output class.
        """

    @abstractmethod
    def get(self, *output: str) -> dict:
        """
        Gets a sub-set of the held data.

        Args:
            *output: Any number of :meth:`fields` elements.

        Returns:
            A sub-set of the held data as a dictionary.
        """


class PropertyOutput(Output):
    """
    :class:`Output` that expects all data to be available as properties (not just
    post-init attributes -- it needs to be real class-level properties so it can be seen
    in :meth:`fields`), and assumes that _all_ non-private attributes (excluding
    :meth:`fields` and :meth:`get` already defined at the parent level) are data.

    Examples)
    >>> from atomistics.shared.output import PropertyOutput
    >>>
    >>> class MyOutput(PropertyOutput):
    ...     def __init__(self, required_data: int):
    ...         self._required_data = required_data
    ...
    ...     @property
    ...     def required_data(self) -> int:
    ...         return self._required_data
    ...
    ...     @property
    ...     def only_got_half(self) -> float:
    ...         return 0.5 * self.required_data
    ('only_got_half', 'required_data')

    >>> mo = MyOutput(42)
    >>> mo.get("only_got_half")
    {'only_got_half': 21.0}
    """
    def get(self, *output: str) -> dict:
        """
        A sub-set of available data as a dictionary.
        """
        return {q: getattr(self, q) for q in output}

    @classmethod
    def fields(cls):
        """
        Names of the data.
        """
        return tuple(
            q for q in dir(cls)
            if not (q[0] == "_" or q in ["get", "fields"])
        )


@dataclasses.dataclass
class EngineOutput(Output):
    """
    :class:`Output` that places an `engine` between the data fields and the dictionary
    returned by :meth:`get`. This allows us to define a single set of fields, but inject
    another class (the `engine`) to modify how these data get calculated.

    Engines provided to :meth:`get` must have properties for each of the data fields.
    """
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    def get(self, engine, *output: str) -> dict:
        return {q: getattr(self, q)(engine) for q in output}


@dataclasses.dataclass
class OutputStatic(EngineOutput):
    forces: callable
    energy: callable
    stress: callable
    volume: callable


@dataclasses.dataclass
class OutputMolecularDynamics(EngineOutput):
    positions: callable
    cell: callable
    forces: callable
    temperature: callable
    energy_pot: callable
    energy_tot: callable
    pressure: callable
    velocities: callable
    volume: callable


@dataclasses.dataclass
class OutputThermalExpansion(EngineOutput):
    temperatures: callable
    volumes: callable


@dataclasses.dataclass
class OutputThermodynamic(OutputThermalExpansion):
    free_energy: callable
    entropy: callable
    heat_capacity: callable


@dataclasses.dataclass
class EquilibriumEnergy(EngineOutput):
    energy_eq: callable


@dataclasses.dataclass
class EquilibriumVolume(EngineOutput):
    volume_eq: callable


@dataclasses.dataclass
class EquilibriumBulkModul(EngineOutput):
    bulkmodul_eq: callable


@dataclasses.dataclass
class EquilibriumBulkModulDerivative(EngineOutput):
    b_prime_eq: callable


@dataclasses.dataclass
class OutputEnergyVolumeCurve(
    EquilibriumEnergy,
    EquilibriumVolume,
    EquilibriumBulkModul,
    EquilibriumBulkModulDerivative,
):
    fit_dict: callable
    energy: callable
    volume: callable


@dataclasses.dataclass
class OutputPhonons(EngineOutput):
    mesh_dict: callable
    band_structure_dict: callable
    total_dos_dict: callable
    dynamical_matrix: callable
    force_constants: callable
