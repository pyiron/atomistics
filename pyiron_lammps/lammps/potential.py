# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import pandas
import shutil
import os
from pyiron_lammps.state.settings import settings
from pyiron_lammps.structure.atoms import Atoms
from typing import List

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


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
    def _get_potential_df(plugin_name, file_name_lst, backward_compatibility_name):
        """

        Args:
            plugin_name (str):
            file_name_lst (set):
            backward_compatibility_name (str):

        Returns:
            pandas.DataFrame:
        """
        env = os.environ
        resource_path_lst = settings.resource_paths
        for conda_var in ["CONDA_PREFIX", "CONDA_DIR"]:
            if conda_var in env.keys():  # support iprpy-data package
                path_to_add = os.path.join(env[conda_var], "share", "iprpy")
                if path_to_add not in resource_path_lst:
                    resource_path_lst += [path_to_add]
        df_lst = []
        for resource_path in resource_path_lst:
            if os.path.exists(os.path.join(resource_path, plugin_name, "potentials")):
                resource_path = os.path.join(resource_path, plugin_name, "potentials")
            if "potentials" in resource_path or "iprpy" in resource_path:
                for path, folder_lst, file_lst in os.walk(resource_path):
                    for periodic_table_file_name in file_name_lst:
                        if (
                            periodic_table_file_name in file_lst
                            and periodic_table_file_name.endswith(".csv")
                        ):
                            df_lst.append(
                                pandas.read_csv(
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
                            )
        if len(df_lst) > 0:
            return pandas.concat(df_lst)
        else:
            raise ValueError("Was not able to locate the potential files.")

    @staticmethod
    def _get_potential_default_df(
        plugin_name,
        file_name_lst={"potentials_vasp_pbe_default.csv"},
        backward_compatibility_name="defaultvasppbe",
    ):
        """

        Args:
            plugin_name (str):
            file_name_lst (set):
            backward_compatibility_name (str):

        Returns:
            pandas.DataFrame:
        """
        for resource_path in settings.resource_paths:
            pot_path = os.path.join(resource_path, plugin_name, "potentials")
            if os.path.exists(pot_path):
                resource_path = pot_path
            if "potentials" in resource_path:
                for path, folder_lst, file_lst in os.walk(resource_path):
                    for periodic_table_file_name in file_name_lst:
                        if (
                            periodic_table_file_name in file_lst
                            and periodic_table_file_name.endswith(".csv")
                        ):
                            return pandas.read_csv(
                                os.path.join(path, periodic_table_file_name),
                                index_col=0,
                            )
                        elif (
                            periodic_table_file_name in file_lst
                            and periodic_table_file_name.endswith(".h5")
                        ):
                            return pandas.read_hdf(
                                os.path.join(path, periodic_table_file_name), mode="r"
                            )
        raise ValueError("Was not able to locate the potential files.")


class LammpsPotential(object):

    """
    This module helps write commands which help in the control of parameters related to the potential used in LAMMPS
    simulations
    """

    def __init__(self, input_file_name=None):
        self._potential = None
        self._attributes = {}
        self._df = None

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_dataframe):
        self._df = new_dataframe
        # ToDo: In future lammps should also support more than one potential file - that is currently not implemented.
        try:
            self.load_string("".join(list(new_dataframe["Config"])[0]))
        except IndexError:
            raise ValueError(
                "Potential not found! "
                "Validate the potential name by self.potential in self.list_potentials()."
            )

    def remove_structure_block(self):
        self.remove_keys(["units"])
        self.remove_keys(["atom_style"])
        self.remove_keys(["dimension"])

    @property
    def files(self):
        if len(self._df["Filename"].values[0]) > 0 and self._df["Filename"].values[
            0
        ] != [""]:
            absolute_file_paths = [
                files for files in list(self._df["Filename"])[0] if os.path.isabs(files)
            ]
            relative_file_paths = [
                files
                for files in list(self._df["Filename"])[0]
                if not os.path.isabs(files)
            ]
            env = os.environ
            resource_path_lst = settings.resource_paths
            for conda_var in ["CONDA_PREFIX", "CONDA_DIR"]:
                if conda_var in env.keys():  # support iprpy-data package
                    path_to_add = settings.convert_path_to_abs_posix(
                        os.path.join(env[conda_var], "share", "iprpy")
                    )
                    if path_to_add not in resource_path_lst:
                        resource_path_lst.append(path_to_add)
            for path in relative_file_paths:
                absolute_file_paths.append(
                    find_potential_file_base(
                        path=path,
                        resource_path_lst=resource_path_lst,
                        rel_path=os.path.join("lammps", "potentials"),
                    )
                )
            if len(absolute_file_paths) != len(list(self._df["Filename"])[0]):
                raise ValueError("Was not able to locate the potentials.")
            else:
                return absolute_file_paths

    def copy_pot_files(self, working_directory):
        if self.files is not None:
            _ = [shutil.copy(path_pot, working_directory) for path_pot in self.files]

    def get_element_lst(self):
        return list(self._df["Species"])[0]

    def _find_line_by_prefix(self, prefix):
        """
        Find a line that starts with the given prefix.  Differences in white
        space are ignored.  Raises a ValueError if not line matches the prefix.

        Args:
            prefix (str): line prefix to search for

        Returns:
            list: words of the matching line

        Raises:
            ValueError: if not matching line was found
        """

        def isprefix(prefix, lst):
            if len(prefix) > len(lst):
                return False
            return all(n == l for n, l in zip(prefix, lst))

        # compare the line word by word to also match lines that differ only in
        # whitespace
        prefix = prefix.split()
        for parameter, value in zip(self._dataset["Parameter"], self._dataset["Value"]):
            words = (parameter + " " + value).strip().split()
            if isprefix(prefix, words):
                return words

        raise ValueError('No line with prefix "{}" found.'.format(" ".join(prefix)))

    def get_element_id(self, element_symbol):
        """
        Return numeric element id for element. If potential does not contain
        the element raise a :class:NameError.  Only makes sense for potentials
        with pair_style "full".

        Args:
            element_symbol (str): short symbol for element

        Returns:
            int: id matching the given symbol

        Raise:
            NameError: if potential does not contain this element
        """

        try:
            line = "group {} type".format(element_symbol)
            return int(self._find_line_by_prefix(line)[3])

        except ValueError:
            msg = "potential does not contain element {}".format(element_symbol)
            raise NameError(msg) from None

    def get_charge(self, element_symbol):
        """
        Return charge for element. If potential does not specify a charge,
        raise a :class:NameError.  Only makes sense for potentials
        with pair_style "full".

        Args:
            element_symbol (str): short symbol for element

        Returns:
            float: charge speicified for the given element

        Raises:
            NameError: if potential does not specify charge for this element
        """

        try:
            line = "set group {} charge".format(element_symbol)
            return float(self._find_line_by_prefix(line)[4])

        except ValueError:
            msg = "potential does not specify charge for element {}".format(
                element_symbol
            )
            raise NameError(msg) from None


class LammpsPotentialFile(PotentialAbstract):
    """
    The Potential class is derived from the PotentialAbstract class, but instead of loading the potentials from a list,
    the potentials are loaded from a file.

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

    def __init__(self, potential_df=None, default_df=None, selected_atoms=None):
        if potential_df is None:
            potential_df = self._get_potential_df(
                plugin_name="lammps",
                file_name_lst={"potentials_lammps.csv"},
                backward_compatibility_name="lammpspotentials",
            )
        super(LammpsPotentialFile, self).__init__(
            potential_df=potential_df,
            default_df=default_df,
            selected_atoms=selected_atoms,
        )

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


def view_potentials(structure: Atoms) -> pandas.DataFrame:
    """
    List all interatomic potentials for the given atomistic structure including all potential parameters.

    To quickly get only the names of the potentials you can use `list_potentials()` instead.

    Args:
        structure (Atoms): The structure for which to get potentials.

    Returns:
        pandas.Dataframe: Dataframe including all potential parameters.
    """
    list_of_elements = set(structure.get_chemical_symbols())
    return LammpsPotentialFile().find(list_of_elements)


def list_potentials(structure: Atoms) -> List[str]:
    """
    List of interatomic potentials suitable for the given atomic structure.

    See `view_potentials` to get more details.

    Args:
        structure (Atoms): The structure for which to get potentials.

    Returns:
        list: potential names
    """
    return list(view_potentials(structure)["Name"].values)