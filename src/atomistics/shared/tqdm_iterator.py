from collections.abc import Iterable, Iterator
from typing import TypeVar

try:
    from tqdm import tqdm  # type: ignore[import-untyped]

    tqdm_available = True
except ImportError:
    tqdm_available = False

T = TypeVar("T")


def get_tqdm_iterator(lst: Iterable[T]) -> Iterator[T]:
    """
    Returns an iterator with tqdm progress bar if tqdm is available, otherwise returns the original list iterator.

    Args:
        lst (list): The list to iterate over.

    Returns:
        Iterator: An iterator with tqdm progress bar if tqdm is available, otherwise the original list iterator.
    """
    if tqdm_available:
        return iter(tqdm(lst))
    return iter(lst)
