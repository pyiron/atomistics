import numpy as np
import spglib
from ase.atoms import Atoms


def ase_to_spglib(structure: Atoms) -> tuple:
    """
    Translate ASE to spglib cell. The format is a tuple of
    (basis vectors, atomic points, types). The implementation here follows
    the doc from this page: https://github.com/spglib/spglib/pull/386/files

    TODO: Optional vectors should be available.
    """
    return (
        np.array(structure.get_cell().T, dtype="double", order="C"),
        np.array(structure.get_scaled_positions(), dtype="double", order="C"),
        np.array(structure.get_atomic_numbers(), dtype="intc"),
    )


def find_symmetry_group_number(struct: tuple) -> int:
    """
    Find the symmetry group number (SGN) of a given structure.

    Parameters:
    struct (tuple): The structure in the format of (basis vectors, atomic points, types).

    Returns:
    int: The symmetry group number (SGN) of the structure.
    """
    dataset = spglib.get_symmetry_dataset(cell=ase_to_spglib(struct))
    SGN = dataset["number"]
    return SGN


# Ref. Com. Phys. Comm. 184 (2013) 1861-1873
# and ElaStic source code
# for more details
# http://exciting-code.org/elastic
Ls_Dic = {
    "01": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    "02": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "03": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "04": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "05": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    "06": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    "07": [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    "08": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "09": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "10": [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    "11": [1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    "12": [1.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    "13": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    "14": [0.0, 1.0, 0.0, 2.0, 0.0, 0.0],
    "15": [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
    "16": [0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
    "17": [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
    "18": [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
    "19": [0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
    "20": [0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
    "21": [0.0, 0.0, 0.0, 2.0, 0.0, 2.0],
    "22": [0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
    "23": [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
    "24": [-1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    "25": [0.5, -1.0, 0.5, 0.0, 0.0, 0.0],
    "26": [0.5, 0.5, -1.0, 0.0, 0.0, 0.0],
    "27": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    "28": [1.0, -1.0, 0.0, 0.0, 0.0, 2.0],
    "29": [0.0, 1.0, -1.0, 0.0, 0.0, 2.0],
    "30": [0.5, 0.5, -1.0, 0.0, 0.0, 2.0],
    "31": [1.0, 0.0, 0.0, 2.0, 2.0, 0.0],
    "32": [1.0, 1.0, -1.0, 0.0, 0.0, 0.0],
    "33": [1.0, 1.0, 1.0, -2.0, -2.0, -2.0],
    "34": [0.5, 0.5, -1.0, 2.0, 2.0, 2.0],
    "35": [0.0, 0.0, 0.0, 2.0, 2.0, 4.0],
    "36": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "37": [-2.0, 1.0, 4.0, -3.0, 6.0, -5.0],
    "38": [3.0, -5.0, -1.0, 6.0, 2.0, -4.0],
    "39": [-4.0, -6.0, 5.0, 1.0, -3.0, 2.0],
    "40": [5.0, 4.0, 6.0, -2.0, -1.0, -3.0],
    "41": [-6.0, 3.0, -2.0, 5.0, -4.0, 1.0],
}


def get_symmetry_family_from_SGN(SGN: int) -> str:
    """
    Get the symmetry family (LC) from the symmetry group number (SGN).

    Parameters:
    SGN (int): The symmetry group number.

    Returns:
    str: The symmetry family (LC).
    """
    if 1 <= SGN <= 2:  # Triclinic
        LC = "N"
    elif 3 <= SGN <= 15:  # Monoclinic
        LC = "M"
    elif 16 <= SGN <= 74:  # Orthorhombic
        LC = "O"
    elif 75 <= SGN <= 88:  # Tetragonal II
        LC = "TII"
    elif 89 <= SGN <= 142:  # Tetragonal I
        LC = "TI"
    elif 143 <= SGN <= 148:  # Rhombohedral II
        LC = "RII"
    elif 149 <= SGN <= 167:  # Rhombohedral I
        LC = "RI"
    elif 168 <= SGN <= 176:  # Hexagonal II
        LC = "HII"
    elif 177 <= SGN <= 194:  # Hexagonal I
        LC = "HI"
    elif 195 <= SGN <= 206:  # Cubic II
        LC = "CII"
    elif 207 <= SGN <= 230:  # Cubic I
        LC = "CI"
    else:
        raise ValueError("SGN should be 1 <= SGN <= 230")
    return LC


def get_LAG_Strain_List(LC: str) -> list[str]:
    """
    Get the Lag_strain_list based on the symmetry family (LC).

    Parameters:
    LC (str): The symmetry family.

    Returns:
    list[str]: The Lag_strain_list.
    """
    if LC in ("CI", "CII"):
        Lag_strain_list = ["01", "08", "23"]
    elif LC in ("HI", "HII"):
        Lag_strain_list = ["01", "26", "04", "03", "17"]
    elif LC == "RI":
        Lag_strain_list = ["01", "08", "04", "02", "05", "10"]
    elif LC == "RII":
        Lag_strain_list = ["01", "08", "04", "02", "05", "10", "11"]
    elif LC == "TI":
        Lag_strain_list = ["01", "26", "27", "04", "05", "07"]
    elif LC == "TII":
        Lag_strain_list = ["01", "26", "27", "28", "04", "05", "07"]
    elif LC == "O":
        Lag_strain_list = ["01", "26", "25", "27", "03", "04", "05", "06", "07"]
    elif LC == "M":
        Lag_strain_list = [
            "01",
            "25",
            "24",
            "28",
            "29",
            "27",
            "20",
            "12",
            "03",
            "04",
            "05",
            "06",
            "07",
        ]
    else:  # (LC == 'N'):
        Lag_strain_list = [
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
        ]
    return Lag_strain_list


def get_C_from_A2(A2: np.ndarray, LC: str) -> np.ndarray:
    C = np.zeros((6, 6))

    # %!%!%--- Cubic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC in ("CI", "CII"):
        C[0, 0] = -2.0 * (A2[0] - 3.0 * A2[1]) / 3.0
        C[1, 1] = C[0, 0]
        C[2, 2] = C[0, 0]
        C[3, 3] = A2[2] / 6.0
        C[4, 4] = C[3, 3]
        C[5, 5] = C[3, 3]
        C[0, 1] = (2.0 * A2[0] - 3.0 * A2[1]) / 3.0
        C[0, 2] = C[0, 1]
        C[1, 2] = C[0, 1]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Hexagonal structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC in ("HI", "HII"):
        C[0, 0] = 2.0 * A2[3]
        C[0, 1] = 2.0 / 3.0 * A2[0] + 4.0 / 3.0 * A2[1] - 2.0 * A2[2] - 2.0 * A2[3]
        C[0, 2] = 1.0 / 6.0 * A2[0] - 2.0 / 3.0 * A2[1] + 0.5 * A2[2]
        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[2, 2] = 2.0 * A2[2]
        C[3, 3] = -0.5 * A2[2] + 0.5 * A2[4]
        C[4, 4] = C[3, 3]
        C[5, 5] = 0.5 * (C[0, 0] - C[0, 1])
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Rhombohedral I structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
    if LC == "RI":
        C[0, 0] = 2.0 * A2[3]
        C[0, 1] = A2[1] - 2.0 * A2[3]
        C[0, 2] = 0.5 * (A2[0] - A2[1] - A2[2])
        C[0, 3] = 0.5 * (-A2[3] - A2[4] + A2[5])
        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[1, 3] = -C[0, 3]
        C[2, 2] = 2.0 * A2[2]
        C[3, 3] = 0.5 * A2[4]
        C[4, 4] = C[3, 3]
        C[4, 5] = C[0, 3]
        C[5, 5] = 0.5 * (C[0, 0] - C[0, 1])
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Rhombohedral II structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC == "RII":
        C[0, 0] = 2.0 * A2[3]
        C[0, 1] = A2[1] - 2.0 * A2[3]
        C[0, 2] = 0.5 * (A2[0] - A2[1] - A2[2])
        C[0, 3] = 0.5 * (-A2[3] - A2[4] + A2[5])
        C[0, 4] = 0.5 * (-A2[3] - A2[4] + A2[6])
        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[1, 3] = -C[0, 3]
        C[1, 4] = -C[0, 4]
        C[2, 2] = 2.0 * A2[2]
        C[3, 3] = 0.5 * A2[4]
        C[3, 5] = -C[0, 4]
        C[4, 4] = C[3, 3]
        C[4, 5] = C[0, 3]
        C[5, 5] = 0.5 * (C[0, 0] - C[0, 1])
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Tetragonal I structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
    if LC == "TI":
        C[0, 0] = (A2[0] + 2.0 * A2[1]) / 3.0 + 0.5 * A2[2] - A2[3]
        C[0, 1] = (A2[0] + 2.0 * A2[1]) / 3.0 - 0.5 * A2[2] - A2[3]
        C[0, 2] = A2[0] / 6.0 - 2.0 * A2[1] / 3.0 + 0.5 * A2[3]
        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[2, 2] = 2.0 * A2[3]
        C[3, 3] = 0.5 * A2[4]
        C[4, 4] = C[3, 3]
        C[5, 5] = 0.5 * A2[5]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Tetragonal II structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC == "TII":
        C[0, 0] = (A2[0] + 2.0 * A2[1]) / 3.0 + 0.5 * A2[2] - A2[4]
        C[1, 1] = C[0, 0]
        C[0, 1] = (A2[0] + 2.0 * A2[1]) / 3.0 - 0.5 * A2[2] - A2[4]
        C[0, 2] = A2[0] / 6.0 - (2.0 / 3.0) * A2[1] + 0.5 * A2[4]
        C[0, 5] = (-A2[2] + A2[3] - A2[6]) / 4.0
        C[1, 2] = C[0, 2]
        C[1, 5] = -C[0, 5]
        C[2, 2] = 2.0 * A2[4]
        C[3, 3] = 0.5 * A2[5]
        C[4, 4] = C[3, 3]
        C[5, 5] = 0.5 * A2[6]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Orthorhombic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
    if LC == "O":
        C[0, 0] = (
            2.0 * A2[0] / 3.0 + 4.0 * A2[1] / 3.0 + A2[3] - 2.0 * A2[4] - 2.0 * A2[5]
        )
        C[0, 1] = 1.0 * A2[0] / 3.0 + 2.0 * A2[1] / 3.0 - 0.5 * A2[3] - A2[5]
        C[0, 2] = (
            1.0 * A2[0] / 3.0
            - 2.0 * A2[1] / 3.0
            + 4.0 * A2[2] / 3.0
            - 0.5 * A2[3]
            - A2[4]
        )
        C[1, 1] = 2.0 * A2[4]
        C[1, 2] = -2.0 * A2[1] / 3.0 - 4.0 * A2[2] / 3.0 + 0.5 * A2[3] + A2[4] + A2[5]
        C[2, 2] = 2.0 * A2[5]
        C[3, 3] = 0.5 * A2[6]
        C[4, 4] = 0.5 * A2[7]
        C[5, 5] = 0.5 * A2[8]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Monoclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
    if LC == "M":
        C[0, 0] = (
            2.0 * A2[0] / 3.0
            + 8.0 * (A2[1] + A2[2]) / 3.0
            - 2.0 * (A2[5] + A2[8] + A2[9])
        )
        C[0, 1] = A2[0] / 3.0 + 4.0 * (A2[1] + A2[2]) / 3.0 - 2.0 * A2[5] - A2[9]
        C[0, 2] = (A2[0] - 4.0 * A2[2]) / 3.0 + A2[5] - A2[8]
        C[0, 5] = (
            -1.0 * A2[0] / 6.0
            - 2.0 * (A2[1] + A2[2]) / 3.0
            + 0.5 * (A2[5] + A2[7] + A2[8] + A2[9] - A2[12])
        )
        C[1, 1] = 2.0 * A2[8]
        C[1, 2] = (
            -4.0 * (2.0 * A2[1] + A2[2]) / 3.0 + 2.0 * A2[5] + A2[8] + A2[9] + A2[12]
        )
        C[1, 5] = (
            -1.0 * A2[0] / 6.0
            - 2.0 * (A2[1] + A2[2]) / 3.0
            - 0.5 * A2[3]
            + A2[5]
            + 0.5 * (A2[7] + A2[8] + A2[9])
        )
        C[2, 2] = 2.0 * A2[9]
        C[2, 5] = (
            -1.0 * A2[0] / 6.0
            + 2.0 * A2[1] / 3.0
            - 0.5 * (A2[3] + A2[4] - A2[7] - A2[8] - A2[9] - A2[12])
        )
        C[3, 3] = 0.5 * A2[10]
        C[3, 4] = 0.25 * (A2[6] - A2[10] - A2[11])
        C[4, 4] = 0.5 * A2[11]
        C[5, 5] = 0.5 * A2[12]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Triclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC == "N":
        C[0, 0] = 2.0 * A2[0]
        C[0, 1] = 1.0 * (-A2[0] - A2[1] + A2[6])
        C[0, 2] = 1.0 * (-A2[0] - A2[2] + A2[7])
        C[0, 3] = 0.5 * (-A2[0] - A2[3] + A2[8])
        C[0, 4] = 0.5 * (-A2[0] + A2[9] - A2[4])
        C[0, 5] = 0.5 * (-A2[0] + A2[10] - A2[5])
        C[1, 1] = 2.0 * A2[1]
        C[1, 2] = 1.0 * (A2[11] - A2[1] - A2[2])
        C[1, 3] = 0.5 * (A2[12] - A2[1] - A2[3])
        C[1, 4] = 0.5 * (A2[13] - A2[1] - A2[4])
        C[1, 5] = 0.5 * (A2[14] - A2[1] - A2[5])
        C[2, 2] = 2.0 * A2[2]
        C[2, 3] = 0.5 * (A2[15] - A2[2] - A2[3])
        C[2, 4] = 0.5 * (A2[16] - A2[2] - A2[4])
        C[2, 5] = 0.5 * (A2[17] - A2[2] - A2[5])
        C[3, 3] = 0.5 * A2[3]
        C[3, 4] = 0.25 * (A2[18] - A2[3] - A2[4])
        C[3, 5] = 0.25 * (A2[19] - A2[3] - A2[5])
        C[4, 4] = 0.5 * A2[4]
        C[4, 5] = 0.25 * (A2[20] - A2[4] - A2[5])
        C[5, 5] = 0.5 * A2[5]
    return C


def symmetry_analysis(
    structure: Atoms, eps_range: float, num_of_point: int
) -> tuple[int, float, str, list[str], np.ndarray]:
    """
    Perform symmetry analysis on a given atomic structure.

    Parameters:
        structure (Atoms): The atomic structure.
        eps_range (float): The range of epsilon values.
        num_of_point (int): The number of points to evaluate.

    Returns:
        Tuple[int, float, str, List[str], np.ndarray]: The symmetry group number, volume, symmetry family,
        list of Lagrangian strain types, and array of epsilon values.
    """
    SGN = find_symmetry_group_number(structure)
    v0 = structure.get_volume()
    LC = get_symmetry_family_from_SGN(SGN)
    Lag_strain_list = get_LAG_Strain_List(LC)
    epss = np.linspace(-eps_range, eps_range, num_of_point)
    return SGN, v0, LC, Lag_strain_list, epss
