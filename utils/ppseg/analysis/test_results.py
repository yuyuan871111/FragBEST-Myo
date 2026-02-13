import pickle

import pandas as pd
from natsort import natsort_keygen


def read_results_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        results = pickle.load(f)
    results = pd.DataFrame(results)
    results.sort_values(by="filename", inplace=True, key=natsort_keygen())
    results.reset_index(drop=True, inplace=True)
    return results
