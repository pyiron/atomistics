from typing import Iterator

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False


def get_tqdm_iterator(lst: list) -> Iterator:
    if tqdm_available:
        return tqdm(lst)
    else:
        return lst
