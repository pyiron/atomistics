"""
A wrapper for mapping between functions that evaluate a single structure to those
that evaluate a task dictionary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atomistics.calculators.interface import TaskEnum, TaskOutputEnum

if TYPE_CHECKING:
    from ase import Atoms

    from atomistics.calculators.interface import (
        ResultsDict,
        SimpleEvaluator,
        TaskDict,
        TaskName,
    )


def _convert_task_dict(old_task_dict: dict[TaskName, dict[str, Atoms]]) -> TaskDict:
    """
    Converts the existing task dictionaries of the format
    `{result_type_string: {structure_label_string: structure, ...}, ...}`
    to the new format
    `{structure_label_string: (structure, [result_type_string, ...]), ...}`.

    Can be removed if/when the rest of the codebase passing in these task
    dictionaries gets updated to the new format.
    """
    task_dict = {}
    for method_name, subdict in old_task_dict.items():
        if not isinstance(subdict, dict):
            subdict = {"label_hidden": subdict}
        for label, structure in subdict.items():
            try:
                task_dict[label][1].append(method_name)
            except KeyError:
                task_dict[label] = (structure, [method_name])
    return task_dict


def as_task_dict_evaluator(
    calculate: SimpleEvaluator,
) -> callable[[dict[TaskName, dict[str, Atoms]], ...], ResultsDict]:
    """
    Takes a callable that acts on a single structure and a (string) list of tasks to
    and maps it to a function that operates on a task-list dictionary of structures,
    structure labels, and the same task list strings. Similarly, maps the output from a
    single dictionary of task-name-related-output-labels to a nested dictionary using
    both the output labels and the structure labels.

    Args:
        calculate [callable]: The function that interprets structures into physical
            properties.

    Returns:
        callable: The function operating on a different space.
    """

    def evaluate_with_calculator(
        task_dict: dict[TaskName, dict[str, Atoms]],
        # TODO: Make workflows pass task dicts: dict[str, TaskSpec] ~ TaskDict,
        *calculate_args,
        **calculate_kwargs,
    ) -> ResultsDict:
        task_dict = _convert_task_dict(task_dict)
        results_dict = {}
        for label, (structure, task_lst) in task_dict.items():
            tasks = [TaskEnum(t) for t in task_lst]
            output = calculate(structure, tasks, *calculate_args, **calculate_kwargs)
            for task_name in tasks:
                result_name = TaskOutputEnum(task_name).name
                if label != "label_hidden":
                    try:
                        results_dict[result_name][label] = output[result_name]
                    except KeyError:
                        results_dict[result_name] = {label: output[result_name]}
                else:
                    results_dict[result_name] = output[result_name]
        return results_dict

    return evaluate_with_calculator
