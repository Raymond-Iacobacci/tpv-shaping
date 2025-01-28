import ast
import json
import os

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import numpy.typing as npt
import pandas as pd  # type: ignore

def home_directory():
    return os.path.expanduser("~/tpv-shaping")

def load_config_in_dir(dir_path):
    """Try to load config.json from dir_path. Return dict if successful, else None."""
    config_path = os.path.join(dir_path, 'config.json')
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def replace_nan_with_neighbors(arr):
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            left = arr[i-1] if i > 0 else arr[i+1]
            right = arr[i+1] if i < len(arr)-1 else arr[i-1]
            arr[i] = (left + right) / 2
    return arr


def quar(mat: npt.ArrayLike) -> np.ndarray:
    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] % 2 == 0
    mid = int(mat.shape[0] / 2)
    return np.array([mat[:mid, :mid], mat[:mid, mid:], mat[mid:, :mid], mat[mid:, mid:]])

# TODO vectorize for GPU use and give errors


def manual_matmul(A, B, threshold=None, dtype=None):
    if dtype is None:
        dtype = np.cdouble
    else:
        dtype = np.dtype(dtype)
    """
    Manually performs matrix multiplication between two NumPy arrays A and B.

    Parameters:
    - A: NumPy array of shape (m, n)
    - B: NumPy array of shape (n, p)

    Returns:
    - result: NumPy array of shape (m, p) resulting from A x B
    """

    # Get the dimensions of the input matrices
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape

    # Check if the matrices can be multiplied
    if a_cols != b_rows:
        raise ValueError("Incompatible dimensions for matrix multiplication.")

    # Initialize the result matrix with zeros
    # This class has worked so far to elim half-zero errors
    result = np.zeros((a_rows, b_cols), dtype=dtype)

    # Perform the matrix multiplication manually
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):  # or range(b_rows)
                result[i, j] += A[i, k] * B[k, j]
    if threshold is not None:
        result[np.abs(result) < threshold] *= 0
    return result


def read_boolean_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        _, boolean_values = line.split(",", 1)
        boolean_list = ast.literal_eval(boolean_values.strip())
        int_list = [int(value) for value in boolean_list]
        data.append(int_list)
    data_array = np.array(data, dtype=int)
    return data_array


def parse_data(lines):
    data_arrays = []
    for line in lines:
        if line.strip():  # This checks if the line is not empty
            array_str = line.split(",", 1)[1].strip().strip("[]")
            array = np.array(list(map(float, array_str.split(","))))
            data_arrays.append(array)
    return np.array(data_arrays)


def read_data(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            value = line.strip().split(", ")[1]
            data.append(float(value))
    return np.array(data, dtype=object)


def plotLens(array):
    plt.figure(figsize=(10, 5))
    plt.step(range(len(array)), array, where="post")
    plt.show()