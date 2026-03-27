from atomistics.referencedata.wikipedia import (
    get_elastic_properties as get_elastic_properties_from_wikipedia,
)
from atomistics.shared.import_warning import raise_warning

__all__: list[str] = ["get_elastic_properties_from_wikipedia"]
data_functions: list[str] = [
    "get_chemical_information_from_mendeleev",
    "get_chemical_information_from_wolframalpha",
]


try:
    from atomistics.referencedata.mendeleev import (
        get_chemical_information as get_chemical_information_from_mendeleev,
    )
    from atomistics.referencedata.wolframalpha import (
        get_chemical_information as get_chemical_information_from_wolframalpha,
    )
except ImportError as e:
    raise_warning(module_list=data_functions, import_error=e)
else:
    __all__ += data_functions
