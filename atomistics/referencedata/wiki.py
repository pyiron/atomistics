import pandas


def get_experimental_elastic_property_wikipedia(chemical_symbol: str) -> dict:
    """
    Looks up elastic properties for a given chemical symbol from the Wikipedia: https://en.wikipedia.org/wiki/Elastic_properties_of_the_elements_(data_page) sourced from webelements.com.

    Args:
        chemical_symbol (str): Chemical symbol of the element.
        property (str): Name of the property to retrieve. Options: youngs_modulus, poissons_ratio, bulk_modulus, shear_modulus

    Returns:
        str: Property value (various types): Value of the property for the given element, if available.
    """
    property_lst = [
        "youngs_modulus",
        "poissons_ratio",
        "bulk_modulus",
        "shear_modulus",
    ]
    df_lst = pandas.read_html(
        "https://en.wikipedia.org/wiki/Elastic_properties_of_the_elements_(data_page)"
    )
    property_dict = {}
    for i, p in enumerate(property_lst):
        df_tmp = df_lst[i]
        property_dict[p] = float(
            df_tmp[df_tmp.symbol == chemical_symbol].squeeze(axis=0).to_dict()["WEL[1]"]
        )
    return property_dict
