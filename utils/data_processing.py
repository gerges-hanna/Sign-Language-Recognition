"""
@author: Gerges_Hanna
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def read_dataset_fromCSV(csv_path, sequence_length):
    """used to read the data from csv file.

    Requirement
    -----------
    Initialize pandas library.
    Initialize numpy library.

    Parameters
    ----------
    csv_path : string
        Take the path of the data.
    sequence_length : int
        Take the number of unified frames for each video.

    Returns
    -------
    The data and the labels for each word.
    """
    getData = pd.read_csv(csv_path)
    labels = getData.label
    getData.drop('label', axis=1, inplace=True)

    # changing the shape, -1 means any number which is suitable
    X = np.array(getData).reshape((len(getData), sequence_length, -1))

    return X, np.array(labels)


def read_meta_data(meta_data_path):
    meta_data=pd.read_csv(meta_data_path)
    meta_data=meta_data.to_dict("split")
    meta_data=meta_data['data']
    meta_data=dict(meta_data)
    return meta_data

def save_meta_data_as_CSV(meta_data:dict,path):
    df = pd.DataFrame(meta_data, index=[0]).transpose()
    df.to_csv(os.path.join(path, "Meta-Data.csv"))


def update_history(history, patience):
    """
      Update the history : Remove the patience epochs from the history to get the final real result

      input
      -----
      history  : history of model

      patience : the patience for early stopping

      output
      ------
      return the updated history

    """
    base_history = history.history
    for i in base_history:
        base_history[i] = base_history[i][0:len(base_history[i]) - patience]
    history.history = base_history
    return history


def split_train_test_validation(X, y, test_size, val_size, random_state=42):
    """used to split the data to train, test and validation.

    Requirement
    -----------
    Initialize numpy library.
    Initialize sklearn.model_selection library.

    Parameters
    ----------
    X : list
        take the data.
    y : list
        take the actions name.
    test_size : float
        take the percentage of data to use in the test.
    val_size : float
        take the percentage of data to use in the validation.

    Returns
    -------
    X_train : list
        data for train.
    X_test : list
        data for test.
    X_val : list
        data for validation.
    y_train : list
        labels for train.
    y_test : list
        labels for test.
    y_val : list
        labels for validation.
    """
    X = np.array(X)
    remain = (test_size + val_size)
    X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=remain, random_state=random_state,
                                                            stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=(val_size / remain),
                                                    random_state=random_state, stratify=y_remain)
    return X_train, X_test, X_val, y_train, y_test, y_val
