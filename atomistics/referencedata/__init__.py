from atomistics.referencedata.wikipedia import (
    get_elastic_properties as get_elastic_properties_from_wikipedia,
)

try:
    from atomistics.referencedata.mendeleev import (
        get_chemical_information as get_chemical_information_from_mendeleev,
    )
    from atomistics.referencedata.wolframalpha import (
        get_chemical_information as get_chemical_information_from_wolframalpha,
    )
except ImportError:
    __all__ = []
else:
    __all__ = [
        "get_chemical_information_from_mendeleev",
        "get_chemical_information_from_wolframalpha",
    ]


__all__ += ["get_elastic_properties_from_wikipedia"]
