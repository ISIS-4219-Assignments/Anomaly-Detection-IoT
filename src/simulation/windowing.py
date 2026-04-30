"""
windowing.py
------------

Converts a 2-D feature matrix into overlapping sliding windows, turning a
flat tabular dataset into a time-series dataset suitable for LSTM and Conv1D
autoencoders.

This step sits between the preprocessor and the model:

    preprocessor.fit_transform()  →  X shape: (N, features)
            ↓
    create_windows(X, window_size)  →  X shape: (N - W + 1, W, features)
            ↓
    model.fit(X, X)

The vanilla autoencoder does NOT use this module (it receives flat vectors).
Set WINDOW_SIZE = None in main.py when MODEL_TYPE = "vanilla".
"""


import numpy as np


def create_windows(X: np.ndarray, window_size: int) -> np.ndarray:

    """Convert a 2-D feature matrix into overlapping sliding windows.

    Each window contains ``window_size`` consecutive rows of ``X``, and
    windows overlap by ``window_size - 1`` rows (stride = 1).  The total
    number of windows is ``N - window_size + 1``.

    Parameters
    ----------
    X : np.ndarray, shape (N, n_features)
        Scaled feature matrix produced by :class:`IoTPreprocessor`.
        Rows must already be in chronological order within each device split.
    window_size : int
        Number of consecutive rows that form one time-series sample.
        For example, ``window_size=30`` means each sample covers 30 network
        traces.

    Returns
    -------
    np.ndarray, shape (N - window_size + 1, window_size, n_features)
        3-D array of overlapping windows, ready to feed directly into a
        Keras LSTM or Conv1D layer.

    Raises
    ------
    ValueError
        If ``window_size`` is larger than the number of rows in ``X``.

    Examples
    --------
    >>> X = np.arange(12).reshape(4, 3).astype('float32')
    >>> create_windows(X, window_size=2).shape
    (3, 2, 3)
    """
    
    n_rows = X.shape[0]

    if window_size > n_rows:
        raise ValueError(
            f"window_size={window_size} is larger than the number of rows "
            f"in X ({n_rows}). Reduce window_size or use a larger split."
        )

    n_windows = n_rows - window_size + 1
    return np.stack([X[i : i + window_size] for i in range(n_windows)])
