import pandas


def get_elastic_properties(chemical_symbol: str) -> dict:
    """
    Looks up elastic properties for a given chemical symbol from the Wikipedia: https://en.wikipedia.org/wiki/Elastic_properties_of_the_elements_(data_page)
    sourced from webelements.com.

    Args:
        chemical_symbol (str): Chemical symbol of the element.

    Returns:
        dict: Dictionary with the following keys
            youngs_modulus: Young's modulus (or Young modulus) in GPa is a mechanical property of solid materials that
                            measures the tensile or compressive stiffness when the force is applied lengthwise.
            poissons_ratio: In materials science and solid mechanics, Poisson's ratio ν (nu) is a measure of the Poisson
                            effect, the deformation (expansion or contraction) of a material in directions perpendicular
                            to the specific direction of loading.
            bulk_modulus: The bulk modulus (K or B or k) in GPa of a substance is a measure of the resistance of a
                          substance to bulk compression.
            shear_modulus: In materials science, shear modulus or modulus of rigidity in GPa, denoted by G, or sometimes
                           S or μ, is a measure of the elastic shear stiffness of a material and is defined as the ratio
                           of shear stress to the shear strain.
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
