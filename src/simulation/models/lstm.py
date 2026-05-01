"""
lstm.py
-------

LSTM sequence autoencoder for time-series anomaly detection.

Architecture
------------
Input  (batch, window, features)
  → Encoder : LSTM(64, return_sequences=True)
              LSTM(32, return_sequences=True)
              LSTM(latent_dim, return_sequences=False)   ← bottleneck
  → Decoder : Dense(window * 32, relu)
              Reshape(window, 32)
              LSTM(64, return_sequences=True)
              TimeDistributed(Dense(features, linear))
  → Output  (batch, window, features)

The encoder collapses the full window into a single latent vector (the last
hidden state of the third LSTM).  The decoder projects that vector back to a
``(window_size, 32)`` sequence via a Dense + Reshape before passing it to the
decoder LSTM, which reconstructs each time step independently.

Input shape  : (batch_size, window_size, n_features)
Output shape : (batch_size, window_size, n_features)
Loss         : Mean Squared Error (MSE) over every time step and feature.
"""


from keras import Input, Model
from keras import layers


def build_lstm(input_dim: int, window_size: int, latent_dim: int = 32) -> Model:

    """Build and compile an LSTM sequence autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of features per time step (channels in the sequence).
    window_size : int
        Number of consecutive rows that form one sample — must match the
        value passed to :func:`windowing.create_windows`.
    latent_dim : int, optional
        Size of the LSTM bottleneck (units in the innermost encoder/decoder
        LSTM layers).  Default: 32.

    Returns
    -------
    keras.Model
        Compiled Keras model.
        - Input shape : ``(batch_size, window_size, input_dim)``
        - Output shape: ``(batch_size, window_size, input_dim)``
        - Optimizer   : Adam (default learning rate)
        - Loss        : Mean Squared Error
    """

    inputs = Input(shape=(window_size, input_dim), name="input")

    # --- Encoder ---
    x = layers.LSTM(64, return_sequences=True, name="enc_lstm_1")(inputs)
    x = layers.LSTM(32, return_sequences=True, name="enc_lstm_2")(x)

    # return_sequences=False collapses the time axis → bottleneck vector
    encoded = layers.LSTM(latent_dim, return_sequences=False, name="bottleneck")(x)

    # --- Decoder ---
    x = layers.Dense(window_size * 32, activation="relu", name = "dec_dense")(encoded)
    x = layers.Reshape((window_size, 32), name = "reshape")(x)
    x = layers.LSTM(64, return_sequences=True, name="dec_lstm_1")(x)

    # TimeDistributed applies the same Dense independently at each time step
    outputs = layers.TimeDistributed(
        layers.Dense(input_dim, activation="linear"), name="output"
    )(x)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    
    return model
