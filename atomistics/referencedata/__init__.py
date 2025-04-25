import warnings

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
except ImportError as e:
    warnings.warn(
        message="get_chemical_information_from_mendeleev() and get_chemical_information_from_wolframalpha() are not available as import failed for "
        + e.msg[2:],
        stacklevel=2,
    )
    __all__ = []
else:
    __all__ = [
        "get_chemical_information_from_mendeleev",
        "get_chemical_information_from_wolframalpha",
    ]


__all__ += ["get_elastic_properties_from_wikipedia"]
