import pandas
from pathlib import Path
import os
from ase.atoms import Atoms


class PotentialAbstract(object):
    """
    The PotentialAbstract class loads a list of available potentials and sorts them. Afterwards the potentials can be
    accessed through:
        PotentialAbstract.<Element>.<Element> or PotentialAbstract.find_potentials_set({<Element>, <Element>}

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

    def __init__(self, potential_df, default_df=None, selected_atoms=None):
        self._potential_df = potential_df
        self._default_df = default_df
        if selected_atoms is not None:
            self._selected_atoms = selected_atoms
        else:
            self._selected_atoms = []

    def find(self, element):
        """
        Find the potentials

        Args:
            element (set, str): element or set of elements for which you want the possible LAMMPS potentials

        Returns:
            list: of possible potentials for the element or the combination of elements

        """
        if isinstance(element, set):
            element = element
        elif isinstance(element, list):
            element = set(element)
        elif isinstance(element, str):
            element = set([element])
        else:
            raise TypeError("Only, str, list and set supported!")
        return self._potential_df[
            [
                True if set(element).issubset(species) else False
                for species in self._potential_df["Species"].values
            ]
        ]

    def find_by_name(self, potential_name):
        mask = self._potential_df["Name"] == potential_name
        if not mask.any():
            raise ValueError(
                "Potential '{}' not found in database.".format(potential_name)
            )
        return self._potential_df[mask]

    def list(self):
        """
        List the available potentials

        Returns:
            list: of possible potentials for the element or the combination of elements
        """
        return self._potential_df

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item):
        potential_df = self.find(element=item)
        selected_atoms = self._selected_atoms + [item]
        return PotentialAbstract(
            potential_df=potential_df,
            default_df=self._default_df,
            selected_atoms=selected_atoms,
        )

    def __str__(self):
        return str(self.list())

    @staticmethod
    def _get_potential_df(file_name_lst, resource_path):
        """

        Args:
            plugin_name (str):
            file_name_lst (set):
            resource_path (str):

        Returns:
            pandas.DataFrame:
        """
        for path, folder_lst, file_lst in os.walk(resource_path):
            for periodic_table_file_name in file_name_lst:
                if (
                    periodic_table_file_name in file_lst
                    and periodic_table_file_name.endswith(".csv")
                ):
                    return pandas.read_csv(
                        os.path.join(path, periodic_table_file_name),
                        index_col=0,
                        converters={
                            "Species": lambda x: x.replace("'", "")
                            .strip("[]")
                            .split(", "),
                            "Config": lambda x: x.replace("'", "")
                            .replace("\\n", "\n")
                            .strip("[]")
                            .split(", "),
                            "Filename": lambda x: x.replace("'", "")
                            .strip("[]")
                            .split(", "),
                        },
                    )
        raise ValueError("Was not able to locate the potential files.")


class LammpsPotentialFile(PotentialAbstract):
    """
    The Potential class is derived from the PotentialAbstract class, but instead of loading the potentials from a list,
    the potentials are loaded from a file.

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

    def __init__(
        self,
        potential_df=None,
        default_df=None,
        selected_atoms=None,
        resource_path=None,
    ):
        if potential_df is None:
            potential_df = self._get_potential_df(
                file_name_lst={"potentials_lammps.csv"},
                resource_path=resource_path,
            )
        super(LammpsPotentialFile, self).__init__(
            potential_df=potential_df,
            default_df=default_df,
            selected_atoms=selected_atoms,
        )
        self._resource_path = resource_path

    def default(self):
        if self._default_df is not None:
            atoms_str = "_".join(sorted(self._selected_atoms))
            return self._default_df[
                (self._default_df["Name"] == self._default_df.loc[atoms_str].values[0])
            ]
        return None

    def find_default(self, element):
        """
        Find the potentials

        Args:
            element (set, str): element or set of elements for which you want the possible LAMMPS potentials
            path (bool): choose whether to return the full path to the potential or just the potential name

        Returns:
            list: of possible potentials for the element or the combination of elements

        """
        if isinstance(element, set):
            element = element
        elif isinstance(element, list):
            element = set(element)
        elif isinstance(element, str):
            element = set([element])
        else:
            raise TypeError("Only, str, list and set supported!")
        element_lst = list(element)
        if self._default_df is not None:
            merged_lst = list(set(self._selected_atoms + element_lst))
            atoms_str = "_".join(sorted(merged_lst))
            return self._default_df[
                (self._default_df["Name"] == self._default_df.loc[atoms_str].values[0])
            ]
        return None

    def __getitem__(self, item):
        potential_df = self.find(element=item)
        selected_atoms = self._selected_atoms + [item]
        return LammpsPotentialFile(
            potential_df=potential_df,
            default_df=self._default_df,
            selected_atoms=selected_atoms,
            resource_path=self._resource_path,
        )


class PotentialAvailable(object):
    def __init__(self, list_of_potentials):
        self._list_of_potentials = {
            "pot_" + v.replace("-", "_").replace(".", "_"): v
            for v in list_of_potentials
        }

    def __getattr__(self, name):
        if name in self._list_of_potentials.keys():
            return self._list_of_potentials[name]
        else:
            raise AttributeError

    def __dir__(self):
        return list(self._list_of_potentials.keys())

    def __repr__(self):
        return str(dir(self))


def find_potential_file_base(path, resource_path_lst, rel_path):
    if path is not None:
        for resource_path in resource_path_lst:
            path_direct = os.path.join(resource_path, path)
            path_indirect = os.path.join(resource_path, rel_path, path)
            if os.path.exists(path_direct):
                return path_direct
            elif os.path.exists(path_indirect):
                return path_indirect
    raise ValueError(
        "Either the filename or the functional has to be defined.",
        path,
        resource_path_lst,
    )


def view_potentials(structure: Atoms, resource_path: str) -> pandas.DataFrame:
    """
    List all interatomic potentials for the given atomistic structure including all potential parameters.

    To quickly get only the names of the potentials you can use `list_potentials()` instead.

    Args:
        structure (Atoms): The structure for which to get potentials.
        resource_path (str): Path to the "lammps/potentials_lammps.csv" file

    Returns:
        pandas.Dataframe: Dataframe including all potential parameters.
    """
    list_of_elements = set(structure.get_chemical_symbols())
    return LammpsPotentialFile(resource_path=resource_path).find(list_of_elements)


def convert_path_to_abs_posix(path: str) -> str:
    """
    Convert path to an absolute POSIX path

    Args:
        path (str): input path.

    Returns:
        str: absolute path in POSIX format
    """
    return (
        Path(path.strip())
        .expanduser()
        .resolve()
        .absolute()
        .as_posix()
        .replace("\\", "/")
    )
