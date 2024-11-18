import hashlib

import numpy as np
import pandas as pd
import random

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    sample_size = int(len(df) * 0.8)
    id_hash = [hashlib.md5(str(df[id_column].iloc[i]).encode("utf-8")).hexdigest() for i in range(len(df))]
    random_id = random.sample(id_hash, sample_size)
    for i in range(len(df)):
        if hashlib.md5(str(df[id_column].iloc[i]).encode("utf-8")).hexdigest() in id_hash:
            df['sample'].iloc[i] = "train"
        else:
            df['sample'].iloc[i] = "test"

    return df
