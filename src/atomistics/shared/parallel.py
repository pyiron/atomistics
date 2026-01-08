from concurrent.futures import Executor


def _convert_task_dict_to_task_lst(task_dict: dict) -> list:
    """
    Convert a task dictionary to a list of tasks.

    Args:
        task_dict (dict): The task dictionary to be converted.

    Returns:
        list: A list of tasks.

    """
    task_lst = []
    for task_name, task_data in task_dict.items():
        if isinstance(task_data, dict):
            for task_parameter, task_object in task_data.items():
                task_lst.append({task_name: {task_parameter: task_object}})
        else:
            task_lst.append({task_name: task_data})
    return task_lst


def _convert_task_lst_to_task_dict(task_lst: list) -> dict:
    """
    Convert a list of tasks into a dictionary representation.

    Args:
        task_lst (list): A list of tasks.

    Returns:
        dict: A dictionary representation of the tasks.

    """
    task_dict = {}
    for task in task_lst:
        for task_name, task_data in task.items():
            if isinstance(task_data, dict):
                if task_name not in task_dict:
                    task_dict[task_name] = {}
                task_dict[task_name].update(dict(task_data.items()))
            else:
                task_dict[task_name] = task_data
    return task_dict


def evaluate_with_parallel_executor(
    evaluate_function: callable, task_dict: dict, executor: Executor, **kwargs
) -> dict:
    """
    Executes the given `evaluate_function` in parallel using the provided `executor` and returns the results as a dictionary.

    Args:
        evaluate_function (callable): The function to be executed in parallel.
        task_dict (dict): A dictionary containing the tasks to be executed.
        executor (Executor): The executor to be used for parallel execution.
        **kwargs: Additional keyword arguments to be passed to the `evaluate_function`.

    Returns:
        dict: A dictionary containing the results of the parallel execution.

    """
    future_lst = [
        executor.submit(evaluate_function, task_dict=task, **kwargs)
        for task in _convert_task_dict_to_task_lst(task_dict=task_dict)
    ]
    return _convert_task_lst_to_task_dict(
        task_lst=[future.result() for future in future_lst]
    )
