import os
from pathlib import Path
from typing import Union

import pandas
from ase.atoms import Atoms

potential_installation = """
Potential installation guide:

1. Check whether iprpy-data is installed. If not, install it using:

`conda install -c conda-forge iprpy-data`

2. Check whether the resource path is set via:

```python
import os
print(os.environ["CONDA_PREFIX"])
```

3. If the resource path is set, you can call the potential using:

```python
from atomistics.calculators import get_potential_by_name


get_potential_by_name(
    potential_name=my_potential,
    resource_path=os.path.join(os.environ["CONDA_PREFIX"], "share", "iprpy"),
)
```

"""


class PotentialAbstract:
    """
    The PotentialAbstract class loads a list of available potentials and sorts them. Afterwards the potentials can be
    accessed through:
        PotentialAbstract.<Element>.<Element> or PotentialAbstract.find_potentials_set({<Element>, <Element>}

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

    def __init__(
        self,
        potential_df: pandas.DataFrame,
        default_df: pandas.DataFrame = None,
        selected_atoms: list[str] = None,
    ):
        self._potential_df = potential_df
        self._default_df = default_df
        if selected_atoms is not None:
            self._selected_atoms = selected_atoms
        else:
            self._selected_atoms = []

    def find(self, element: Union[set[str], list[str], str]) -> pandas.DataFrame:
        """
        Find the potentials

        Args:
            element (set, str): element or set of elements for which you want the possible LAMMPS potentials

        Returns:
            list: of possible potentials for the element or the combination of elements

        """
        if isinstance(element, list):
            element = set(element)
        elif isinstance(element, str):
            element = {element}
        elif not isinstance(element, set):
            raise TypeError("Only, str, list and set supported!")
        return self._potential_df[
            [
                bool(set(element).issubset(species))
                for species in self._potential_df["Species"].values
            ]
        ]

    def find_by_name(self, potential_name: str) -> pandas.DataFrame:
        mask = self._potential_df["Name"] == potential_name
        if not mask.any():
            raise ValueError(f"Potential '{potential_name}' not found in database.")
        return self._potential_df[mask]

    def list(self) -> pandas.DataFrame:
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

    def __str__(self) -> str:
        return str(self.list())

    @staticmethod
    def _get_potential_df(file_name_lst, resource_path):
        """

        Args:
            file_name_lst (set):
            resource_path (str):

        Returns:
            pandas.DataFrame:
        """
        for path, _folder_lst, file_lst in os.walk(resource_path):
            for periodic_table_file_name in file_name_lst:
                if (
                    periodic_table_file_name in file_lst
                    and periodic_table_file_name.endswith(".csv")
                ):
                    return pandas.read_csv(
                        os.path.join(path, periodic_table_file_name),
                        index_col=0,
                        converters={
                            "Species": (
                                lambda x: x.replace("'", "").strip("[]").split(", ")
                            ),
                            "Config": (
                                lambda x: x.replace("'", "")
                                .replace("\\n", "\n")
                                .strip("[]")
                                .split(", ")
                            ),
                            "Filename": (
                                lambda x: x.replace("'", "").strip("[]").split(", ")
                            ),
                        },
                    )
        raise ValueError(
            "Was not able to locate the potential files." + potential_installation
        )


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
        super().__init__(
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

    def find_default(
        self, element: Union[set[str], list[str], str]
    ) -> pandas.DataFrame:
        """
        Find the potentials

        Args:
            element (set, str): element or set of elements for which you want the possible LAMMPS potentials
            path (bool): choose whether to return the full path to the potential or just the potential name

        Returns:
            list: of possible potentials for the element or the combination of elements

        """
        if isinstance(element, list):
            element = set(element)
        elif isinstance(element, str):
            element = {element}
        elif not isinstance(element, set):
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


class PotentialAvailable:
    def __init__(self, list_of_potentials):
        self._list_of_potentials = {
            "pot_" + v.replace("-", "_").replace(".", "_"): v
            for v in list_of_potentials
        }

    def __getattr__(self, name):
        if name in self._list_of_potentials:
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


def update_potential_paths(
    df_pot: pandas.DataFrame, resource_path: str
) -> pandas.DataFrame:
    config_lst = []
    for row in df_pot.itertuples():
        potential_file_lst = row.Filename
        potential_file_path_lst = [
            os.path.join(resource_path, f) for f in potential_file_lst
        ]
        potential_dict = {os.path.basename(f): f for f in potential_file_path_lst}
        potential_commands = []
        for line in row.Config:
            line = line.replace("\n", "")
            for key, value in potential_dict.items():
                line = line.replace(key, value)
            potential_commands.append(line)
        config_lst.append(potential_commands)
    df_pot["Config"] = config_lst
    return df_pot


def get_resource_path_from_conda(
    env_variables: tuple[str] = ("CONDA_PREFIX", "CONDA_DIR"),
) -> str:
    env = os.environ
    for conda_var in env_variables:
        if conda_var in env:
            resource_path = os.path.join(env[conda_var], "share", "iprpy")
            if os.path.exists(resource_path):
                return resource_path
    raise ValueError("No resource_path found" + potential_installation)


def get_potential_dataframe(structure: Atoms, resource_path=None):
    if resource_path is None:
        resource_path = get_resource_path_from_conda()
    return update_potential_paths(
        df_pot=view_potentials(structure=structure, resource_path=resource_path),
        resource_path=resource_path,
    )


def get_potential_by_name(potential_name: str, resource_path=None):
    if resource_path is None:
        resource_path = get_resource_path_from_conda()
    df = LammpsPotentialFile(resource_path=resource_path).list()
    return update_potential_paths(
        df_pot=df[df.Name == potential_name], resource_path=resource_path
    ).iloc[0]


def validate_potential_dataframe(
    potential_dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    if isinstance(potential_dataframe, pandas.Series):
        return potential_dataframe
    elif isinstance(potential_dataframe, pandas.DataFrame):
        if len(potential_dataframe) == 1:
            return potential_dataframe.iloc[0]
        elif len(potential_dataframe) == 0:
            raise ValueError(
                "The potential_dataframe is an empty pandas.DataFrame:",
                potential_dataframe,
            )
        else:
            raise ValueError(
                "The potential_dataframe contains more than one interatomic potential, please select one:",
                potential_dataframe,
            )
    else:
        raise TypeError(
            "The potential_dataframe should be a pandas.DataFrame or pandas.Series, but instead it is of type:",
            type(potential_dataframe),
        )
