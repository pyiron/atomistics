from ase.atoms import Atoms


def optimize_positions_and_volume(structure: Atoms) -> dict:
    return {"optimize_positions_and_volume": structure}


def optimize_positions(structure: Atoms) -> dict:
    return {"optimize_positions": structure}
