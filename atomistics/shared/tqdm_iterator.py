try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False


def get_tqdm_iterator(lst):
    if tqdm_available:
        return tqdm(lst)
    else:
        return lst
