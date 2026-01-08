from ase.atoms import Atoms


def optimize_positions_and_volume(structure: Atoms) -> dict:
    """
    Optimize the positions and volume of the given structure.

    Parameters:
        structure (Atoms): The structure to be optimized.

    Returns:
        dict: A dictionary containing the optimized structure.

    """
    return {"optimize_positions_and_volume": structure}


def optimize_volume(structure: Atoms) -> dict:
    """
    Optimize the volume of the given structure.

    Parameters:
        structure (Atoms): The structure to be optimized.

    Returns:
        dict: A dictionary containing the optimized structure.

    """
    return {"optimize_volume": structure}


def optimize_cell(structure: Atoms) -> dict:
    """
    Optimize the cell of the given structure.

    Parameters:
        structure (Atoms): The structure to be optimized.

    Returns:
        dict: A dictionary containing the optimized structure.

    """
    return {"optimize_cell": structure}


def optimize_positions(structure: Atoms) -> dict:
    """
    Optimize the positions of the given structure.

    Parameters:
        structure (Atoms): The structure to be optimized.

    Returns:
        dict: A dictionary containing the optimized structure.

    """
    return {"optimize_positions": structure}
