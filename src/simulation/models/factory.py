"""
factory.py
----------
Single entry point for model instantiation.

Changing ``MODEL_TYPE`` in ``main.py`` is the *only* change needed to switch
between architectures.  All other files (``device.py``, ``server.py``) call
:func:`build_model` and remain completely unaware of which architecture is
in use.

Supported model types
---------------------
- ``"vanilla"`` : Dense autoencoder.  No windowing required.
- ``"lstm"``    : LSTM sequence autoencoder.  Requires ``window_size``.
- ``"conv1d"``  : 1-D Conv autoencoder.  Requires ``window_size``.
"""

from keras import Model

from .vanilla import build_vanilla
from .lstm import build_lstm
from .conv1d import build_conv1d


AVAILABLE_MODELS = ("vanilla", "lstm", "conv1d")


def build_model(
    model_type: str,
    input_dim: int,
    window_size: int | None = None,
) -> Model:
    """Instantiate and compile a Keras autoencoder by name.

    Parameters
    ----------
    model_type : str
        Architecture to build.  One of ``"vanilla"``, ``"lstm"``,
        ``"conv1d"``.
    input_dim : int
        Number of features after preprocessing.  For sequence models this
        is the number of features *per time step* (not the flattened size).
    window_size : int or None
        Length of the sliding window used in :func:`windowing.create_windows`.
        Required for ``"lstm"`` and ``"conv1d"``; ignored for ``"vanilla"``.
        Should be ``None`` when ``model_type="vanilla"``.

    Returns
    -------
    keras.Model
        Compiled Keras model ready for :meth:`~keras.Model.fit`.

    Raises
    ------
    ValueError
        If ``model_type`` is not recognised, or if ``window_size`` is
        ``None`` when required.

    Examples
    --------
    >>> model = build_model("vanilla", input_dim=64)
    >>> model = build_model("lstm",    input_dim=64, window_size=30)
    >>> model = build_model("conv1d",  input_dim=64, window_size=30)
    """
    if model_type == "vanilla":
        return build_vanilla(input_dim)

    if model_type in ("lstm", "conv1d"):
        if window_size is None:
            raise ValueError(
                f"window_size is required for model_type='{model_type}'. "
                "Set WINDOW_SIZE in main.py to an integer (e.g. 30)."
            )
        if model_type == "lstm":
            return build_lstm(input_dim, window_size)
        return build_conv1d(input_dim, window_size)

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        f"Choose one of {AVAILABLE_MODELS}."
    )
