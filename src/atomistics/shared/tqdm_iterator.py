from collections.abc import Iterator

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False


def get_tqdm_iterator(lst: list) -> Iterator:
    """
    Returns an iterator with tqdm progress bar if tqdm is available, otherwise returns the original list iterator.

    Args:
        lst (list): The list to iterate over.

    Returns:
        Iterator: An iterator with tqdm progress bar if tqdm is available, otherwise the original list iterator.
    """
    if tqdm_available:
        return tqdm(lst)
    else:
        return lst
