"""
vanilla.py
----------

Dense (fully-connected) autoencoder for tabular anomaly detection.

Architecture
------------
Input  → Encoder (Dense 64 → 32 → latent_dim)
       → Decoder (Dense 32 → 64 → input_dim)  → Output

The model is trained as an autoencoder (target == input) on *normal-only*
traffic so that anomalous traces produce a high reconstruction error at
inference time.

Input shape  : (batch_size, n_features)   — flat, no temporal dimension.
Output shape : (batch_size, n_features)
Loss         : Mean Squared Error (MSE)
"""


from keras import Input, Model
from keras import layers


def build_vanilla(input_dim: int, latent_dim: int = 16) -> Model:

    """Build and compile a vanilla dense autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of input features (size of the flat feature vector produced
        by :class:`IoTPreprocessor`).
    latent_dim : int, optional
        Dimensionality of the bottleneck layer.  Smaller values force more
        compression; larger values preserve more information.  Default: 16.

    Returns
    -------
    keras.Model
        Compiled Keras model.
        - Input shape : ``(batch_size, input_dim)``
        - Output shape: ``(batch_size, input_dim)``
        - Optimizer   : Adam (default learning rate)
        - Loss        : Mean Squared Error
    """

    inputs = Input(shape=(input_dim,), name="input")

    # --- Encoder ---
    x = layers.Dense(64, activation="relu", name="enc_1")(inputs)
    x = layers.Dense(32, activation="relu", name="enc_2")(x)
    encoded = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    # --- Decoder ---
    x = layers.Dense(32, activation="relu", name="dec_1")(encoded)
    x = layers.Dense(64, activation="relu", name="dec_2")(x)
    outputs = layers.Dense(input_dim, activation="linear", name="output")(x)

    model = Model(inputs, outputs, name="vanilla_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    
    return model
