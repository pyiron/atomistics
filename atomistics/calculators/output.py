import dataclasses


@dataclasses.dataclass
class AtomisticsOutput:
    @classmethod
    def fields(cls):
        return tuple(field.name for field in dataclasses.fields(cls))

    @classmethod
    def get(cls, engine, *quantities: str) -> dict:
        return {q: getattr(cls, q)(engine) for q in quantities}