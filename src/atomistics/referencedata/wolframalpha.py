import os
from typing import Callable, Optional, Union

import numpy as np
import pandas
import requests
from mendeleev.fetch import fetch_table


def _get_content_from_url(url: str) -> pandas.DataFrame:
    """
    Retrieves content from the given URL and returns a pandas DataFrame.

    Args:
        url (str): The URL to retrieve content from.

    Returns:
        pandas.DataFrame: The content retrieved from the URL as a pandas DataFrame.
    """
    content = pandas.read_html(requests.get(url).text)
    if len(content[8]) > 1:
        return content[8]
    else:
        return content[9]


def _select_function_density(v: str) -> float:
    """
    Convert the given density value to a float.

    Parameters:
        v (str): The density value to be converted.

    Returns:
        float: The converted density value.
    """
    if "g/l" in v:
        return float(v.split()[0]) * 0.001
    else:
        return float(v.split()[0])


def _select_function_split(v: Union[str, float]) -> float:
    """
    Splits a string and returns the first element as a float.

    Args:
        v (Union[str, float]): The input value to be processed.

    Returns:
        float: The first element of the input string as a float, or the input value if it's already a float.
    """
    if isinstance(v, str):
        return float(v.split()[0])
    else:
        return v


def _select_function_lattice(v: str) -> tuple[float, float, float]:
    """
    Convert a string representation of a lattice vector to a tuple of floats.

    Args:
        v (str): The string representation of the lattice vector.

    Returns:
        tuple[float, float, float]: The lattice vector as a tuple of floats.
    """
    return (
        float(v.split(", ")[0]),
        float(v.split(", ")[1]),
        float(v.split(", ")[2].split()[0]),
    )


def _select_function_scientific(v: Union[str, float]) -> float:
    """
    Converts a scientific notation string to a float.

    Args:
        v (Union[str, float]): The value to be converted. If it is a string, it should be in scientific notation.

    Returns:
        float: The converted value.

    Example:
        >>> _select_function_scientific('1.23E-4')
        0.000123
    """
    if isinstance(v, str):
        return float(v.split()[0].replace("×10", "E"))
    else:
        return v


def _extract_entry(df: pandas.DataFrame, n: str) -> Optional[Union[str, float]]:
    """
    Extracts the value corresponding to the given element name from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to extract data from.
        n (str): The element name to extract data for.

    Returns:
        Optional[Union[str, float]]: The extracted value if found, None otherwise.
    """
    for i in range(int(len(df.columns) / 2)):
        if n in df[i].values:
            v = df[i + 1].values[df[i].values.tolist().index(n)]
            return v
    return None


def _default_filter(v: Union[str, float]) -> bool:
    """
    Default filter function to check if the value is valid.

    Args:
        v (Union[str, float]): The value to be checked.

    Returns:
        bool: True if the value is valid, False otherwise.
    """
    return (isinstance(v, str) and "N/A[note]" not in v) or isinstance(v, float)


def _poisson_filter(v: float) -> bool:
    """
    Check if the given value is not None and not NaN.

    Parameters:
        v (float): The value to be checked.

    Returns:
        bool: True if the value is not None and not NaN, False otherwise.
    """
    return v is not None and not np.isnan(v)


def _select_function_poisson(v: Union[str, float]) -> Union[str, float]:
    """
    Selects the Poisson ratio value from a DataFrame entry.

    Args:
        v (Union[str, float]): The value to be selected.

    Returns:
        Union[str, float]: The selected Poisson ratio value.

    """
    return v


def _select_function_mass(v: Union[str, float]) -> Union[float, str]:
    """
    Selects the mass value from a DataFrame entry.

    Args:
        v (Union[str, float]): The value to be selected.

    Returns:
        Union[float, str]: The selected mass value.

    """
    if isinstance(v, str):
        return float(v.replace("[note]", ""))
    else:
        return v


def _extract_lst(
    df: pandas.DataFrame,
    column: str,
    select_function: Callable,
    current_filter: Callable,
) -> tuple[list[str], list[str]]:
    """
    Extracts element and property lists from a DataFrame based on the given column, select function, and current filter.

    Args:
        df (DataFrame): The DataFrame to extract data from.
        column (str): The column name to extract data from.
        select_function (Callable): A function that selects a property value from a DataFrame entry.
        current_filter (Callable): A function that filters DataFrame entries based on a condition.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the element list and property list.

    Raises:
        ValueError: If an error occurs during the extraction process.
    """
    element_lst, property_lst = [], []
    try:
        ptable = fetch_table("elements")
        for n, el in zip(ptable.name.values, ptable.symbol.values):
            v = _extract_entry(df=df, n=n)
            if current_filter(v):
                element_lst.append(el)
                property_lst.append(select_function(v))
    except ValueError as e:
        raise ValueError(
            f"Error extracting {column} for element {el} with value {v}"
        ) from e
    return element_lst, property_lst


def _collect(
    url: str, column: str, select_function: callable, current_filter: callable
) -> pandas.DataFrame:
    """
    Collect data from a given URL and extract specific properties.

    Args:
        url: The URL to fetch the data from.
        column: The name of the column to extract.
        select_function: The function to apply to the extracted data.
        current_filter: The filter function to apply to the extracted data.

    Returns:
        A pandas DataFrame containing the extracted data.
    """
    return pandas.DataFrame(
        dict(
            zip(
                ["element", column],
                _extract_lst(
                    df=_get_content_from_url(url=url),
                    column=column,
                    select_function=select_function,
                    current_filter=current_filter,
                ),
            )
        )
    )


def _get_volume(
    lat_lst: Union[float, tuple[float, float, float]], crystal: str
) -> Optional[float]:
    """
    Calculate the volume of a crystal unit cell.

    Args:
        lat_lst: Lattice constants of the crystal unit cell.
        crystal: Basic crystal lattice structure.

    Returns:
        The volume of the crystal unit cell in cubic meters, or None if the input is invalid.
    """
    if not isinstance(lat_lst, float) and len(lat_lst) == 3:
        if crystal in ["Face-centered Cubic", "Body-centered Cubic"]:
            return lat_lst[0] * lat_lst[1] * lat_lst[2] / 100 / 100 / 100
        elif crystal == "Simple Hexagonal":
            return (
                lat_lst[0]
                * lat_lst[1]
                * lat_lst[2]
                * 3
                * np.sqrt(3)
                / 2
                / 100
                / 100
                / 100
            )
        else:
            return None
    else:
        return None


def _wolframalpha_download() -> None:
    """
    Downloads data from Wolfram Alpha and saves it to a CSV file.

    This function downloads various properties data from Wolfram Alpha using the provided URLs in the `data_dict`.
    The downloaded data is then processed and saved to a CSV file named "wolfram.csv" in the "data" directory.

    Note: This function requires internet connectivity to download the data.
    """
    data_dict = {
        "thermalcondictivity": {
            "url": "https://periodictable.com/Properties/A/ThermalConductivity.an.html",
            "select_function": _select_function_split,
            "current_filter": _default_filter,
        },
        "atomicradius": {
            "url": "https://periodictable.com/Properties/A/AtomicRadius.an.html",
            "select_function": _select_function_split,
            "current_filter": _default_filter,
        },
        "bulkmodulus": {
            "url": "https://periodictable.com/Properties/A/BulkModulus.an.html",
            "select_function": _select_function_split,
            "current_filter": _default_filter,
        },
        "shearmodulus": {
            "url": "https://periodictable.com/Properties/A/ShearModulus.an.html",
            "select_function": _select_function_split,
            "current_filter": _default_filter,
        },
        "youngmodulus": {
            "url": "https://periodictable.com/Properties/A/YoungModulus.an.html",
            "select_function": _select_function_split,
            "current_filter": _default_filter,
        },
        "poissonratio": {
            "url": "https://periodictable.com/Properties/A/PoissonRatio.an.html",
            "select_function": _select_function_poisson,
            "current_filter": _poisson_filter,
        },
        "density": {
            "url": "https://periodictable.com/Properties/A/Density.an.html",
            "select_function": _select_function_density,
            "current_filter": _default_filter,
        },
        "liquiddensity": {
            "url": "https://periodictable.com/Properties/A/LiquidDensity.an.html",
            "select_function": _select_function_split,
            "current_filter": _default_filter,
        },
        "thermalexpansion": {
            "url": "https://periodictable.com/Properties/A/ThermalExpansion.an.html",
            "select_function": _select_function_scientific,
            "current_filter": _default_filter,
        },
        "meltingpoint": {
            "url": "https://periodictable.com/Properties/A/AbsoluteMeltingPoint.an.html",
            "select_function": _select_function_scientific,
            "current_filter": _default_filter,
        },
        "vaporizationheat": {
            "url": "https://periodictable.com/Properties/A/VaporizationHeat.an.html",
            "select_function": _select_function_split,
            "current_filter": _default_filter,
        },
        "specificheat": {
            "url": "https://periodictable.com/Properties/A/SpecificHeat.an.html",
            "select_function": _select_function_split,
            "current_filter": _default_filter,
        },
        "latticeconstant": {
            "url": "https://periodictable.com/Properties/A/LatticeConstants.an.html",
            "select_function": _select_function_lattice,
            "current_filter": _default_filter,
        },
        "crystal": {
            "url": "https://periodictable.com/Properties/A/CrystalStructure.an.html",
            "select_function": _select_function_poisson,
            "current_filter": _default_filter,
        },
        "volmolar": {
            "url": "https://periodictable.com/Properties/A/MolarVolume.an.html",
            "select_function": _select_function_scientific,
            "current_filter": _default_filter,
        },
        "mass": {
            "url": "https://periodictable.com/Properties/A/AtomicMass.an.html",
            "select_function": _select_function_mass,
            "current_filter": _default_filter,
        },
    }
    result = pandas.concat(
        [
            _collect(
                url=v["url"],
                column=k,
                select_function=v["select_function"],
                current_filter=v["current_filter"],
            ).set_index("element")
            for k, v in data_dict.items()
        ],
        axis=1,
        sort=False,
    )
    result["volume"] = result.apply(
        lambda x: _get_volume(lat_lst=x.latticeconstant, crystal=x.crystal), axis=1
    )
    data_path = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_path, exist_ok=True)
    result.to_csv(os.path.join(data_path, "wolfram.csv"))


def get_chemical_information(chemical_symbol: str) -> dict:
    """
    Get information of a given chemical element
    Args:
        chemical_element: Chemical Element like Au for Gold
    Returns:
        dict: Dictionary with the following keys
            element: chemical element
            thermalcondictivity: thermal conductivity
            atomicradius: calculated distance from nucleus of outermost electron
            bulkmodulus: bulk modulus (incompressibility)
            shearmodulus: shear modulus of solid
            youngmodulus: Young's modulus of solid
            poissonratio: Poisson ratio of solid
            density: density at standard temperature and pressure
            liquiddensity: liquid density at melting point
            thermalexpansion: linear thermal expansion coefficient
            meltingpoint: melting temperature in kelvin
            vaporizationheat: latent heat for liquid-gas transition
            specificheat: specific heat capacity
            latticeconstant: crystal lattice constants
            crystal: basic crystal lattice structure
            volmolar: molar volume
            mass: average atomic weight in atomic mass units
            volume: Volume
    """
    filename = os.path.join(os.path.dirname(__file__), "data", "wolfram.csv")
    if not os.path.exists(filename):
        _wolframalpha_download()
    df = pandas.read_csv(filename)
    return df[df.element == chemical_symbol].squeeze(axis=0).to_dict()
