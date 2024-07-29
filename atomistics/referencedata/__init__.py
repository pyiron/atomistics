from atomistics.referencedata.wiki import get_experimental_elastic_property_wikipedia

try:
    from atomistics.referencedata.mendeleevdb import get_chemical_information_from_mendeleev, get_chemical_information_from_wolframalpha
except ImportError:
    __all__ = []
else:
    __all__ = [get_chemical_information_from_mendeleev, get_chemical_information_from_wolframalpha]


__all__ += [get_experimental_elastic_property_wikipedia]

