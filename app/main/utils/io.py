"""IO Utilities.

Contains logic for IO
"""
import json
import pickle
import re

import kaldi_io
import pandas as pd


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_pickle_file(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def read_json_file(file_path):
    """Load json file from disk.

    Args:
        file_path (string): path to file on disk

    Returns:
        dict: serialized json file to python dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def read_file(file_path, mode='r'):
    """Read a file from disk.

    Configurable mode for binary files.

    Args:
        file_path (string): path to file on disk to read
        mode (string): configurable for binary files i.e. 'rb'

    Returns:
        string: contents of the file
    """
    with open(file_path, mode) as f:
        return f.read()


def read_matrix_ark(f):
    """Read an ark file and return the matrix inside.

    Args:
        f (file): python file object

    Returns:
        numpy array: matrix object
    """
    for key, matrix in kaldi_io.read_mat_ark(f):
        return matrix


def read_ark_file(file_path):
    """Read an ark file from disk and return matrix from inside.

    Args:
        file_path (string): path to file on disk

    Returns:
        numpy array: matrix object from ark file
    """
    with open(file_path, 'rb') as f:
        return read_matrix_ark(f)


def read_csv_file(file_path):
    return pd.read_csv(file_path)


def read_custom_csv_file(file_path, columns, regex, process_line_data):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line_data = re.match(regex, line).groups()
            data.append(process_line_data(list(line_data)))

    return pd.DataFrame(data=data, columns=columns)
