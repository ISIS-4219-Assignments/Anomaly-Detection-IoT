"""
conv1d.py
---------

1-D convolutional autoencoder for time-series anomaly detection.

Architecture
------------
  → Input   (batch, window, features)
  → Encoder : Conv1D(32, k=3, same, relu)
              Conv1D(16, k=3, same, relu)
              GlobalAveragePooling1D             ← pools time axis → (batch, 16)
              Dense(latent_dim, relu)            ← bottleneck
  → Decoder : Dense(window * 16, relu)
              Reshape(window, 16)
              Conv1D(32,       k=3, same, relu)
              Conv1D(features, k=3, same, linear)
  → Output  (batch, window, features)

Both encoder and decoder Conv1D layers use ``padding='same'`` so the temporal
length (window_size) is preserved throughout — no strided downsampling.
GlobalAveragePooling1D collapses the time axis cleanly into a 2-D tensor,
avoiding the awkward 4-D reshape that Flatten produces after Conv1D and that
can trigger XLA tiling failures on newer GPU architectures.

Input shape  : (batch_size, window_size, n_features)
Output shape : (batch_size, window_size, n_features)
Loss         : Mean Squared Error (MSE) over every time step and feature.
"""


from keras import Input, Model
from keras import layers


def build_conv1d(input_dim: int, window_size: int, latent_dim: int = 16) -> Model:

    """Build and compile a 1-D convolutional autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of features per time step (channels in the sequence).
    window_size : int
        Number of consecutive rows that form one sample — must match the
        value passed to :func:`windowing.create_windows`.
    latent_dim : int, optional
        Size of the Dense bottleneck between the convolutional encoder and
        decoder.  Default: 32.

    Returns
    -------
    keras.Model
        Compiled Keras model.
        - Input shape : ``(batch_size, window_size, input_dim)``
        - Output shape: ``(batch_size, window_size, input_dim)``
        - Optimizer   : Adam (default learning rate)
        - Loss        : Mean Squared Error

    Notes
    -----
    The encoder uses ``GlobalAveragePooling1D`` instead of ``Flatten`` to
    collapse the time axis.  This produces a clean ``(batch, 16)`` tensor
    instead of a ``(batch, window * 16)`` tensor, which sidesteps XLA tiling
    failures on newer GPU architectures (e.g. compute capability 12.0+).
    The decoder's Dense layer projects back to ``window_size * 16`` units and
    reshapes to ``(window_size, 16)`` before the decoder Conv1D layers.
    """

    inputs = Input(shape=(window_size, input_dim), name="input")

    # --- Encoder ---
    x = layers.Conv1D(64, kernel_size = 3, padding = "same", activation = "relu", name = "enc_conv_1")(inputs)
    x = layers.Conv1D(32, kernel_size = 3, padding = "same", activation = "relu", name = "enc_conv_2")(x)
    # GlobalAveragePooling1D averages across the time axis → (batch, 16)
    # This avoids the (batch, window*16) shape from Flatten that causes XLA tiling issues
    x = layers.GlobalAveragePooling1D(name = "gap")(x)
    encoded = layers.Dense(latent_dim, activation = "relu", name = "bottleneck")(x)

    # --- Decoder ---
    # Project back to (window_size, 32) to feed the decoder Conv1D layers
    x = layers.Dense(window_size * 32, activation = "relu", name = "dec_dense")(encoded)
    x = layers.Reshape((window_size, 32), name = "reshape")(x)
    x = layers.Conv1D(64, kernel_size = 3, padding = "same", activation = "relu", name = "dec_conv_1")(x)
    outputs = layers.Conv1D(input_dim, kernel_size = 3, padding = "same", activation = "linear", name = "output")(x)

    model = Model(inputs, outputs, name="conv1d_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    
    return model
