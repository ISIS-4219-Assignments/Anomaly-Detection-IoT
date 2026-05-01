"""
transformer.py
--------------

Transformer-based autoencoder for time-series anomaly detection.

Architecture
------------
Input  (batch, window, features)
  → Encoder : Dense(d_model)                        ← feature projection
              MultiHeadAttention + LayerNorm         ← self-attention block
              FFN(ff_dim → d_model) + LayerNorm      ← feed-forward block
              GlobalAveragePooling1D                 ← collapse time axis
              Dense(latent_dim, relu)                ← bottleneck
  → Decoder : Dense(window * d_model, relu)
              Reshape(window, d_model)               ← broadcast back to sequence
              MultiHeadAttention + LayerNorm         ← self-attention block
              FFN(ff_dim → d_model) + LayerNorm      ← feed-forward block
              TimeDistributed(Dense(features, linear))
  → Output  (batch, window, features)

The encoder projects each time step into a ``d_model``-dimensional space,
applies one transformer block (self-attention + FFN with residual connections
and LayerNorm), then pools across the time axis to form a latent vector.
The decoder projects the latent vector back to a sequence of length
``window_size`` and applies a symmetric transformer block before the output
projection.

Input shape  : (batch_size, window_size, n_features)
Output shape : (batch_size, window_size, n_features)
Loss         : Mean Squared Error (MSE) over every time step and feature.
"""


from keras import Input, Model
from keras import layers


def build_transformer(
    input_dim: int,
    window_size: int,
    latent_dim: int = 16,
    num_heads: int = 4,
    d_model: int = 64,
    ff_dim: int = 128,
) -> Model:

    """Build and compile a Transformer sequence autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of features per time step (channels in the sequence).
    window_size : int
        Number of consecutive rows that form one sample — must match the
        value passed to :func:`windowing.create_windows`.
    latent_dim : int, optional
        Size of the Dense bottleneck that separates encoder from decoder.
        Default: 16.
    num_heads : int, optional
        Number of attention heads in both MHA layers.  Must evenly divide
        ``d_model``.  Default: 4.
    d_model : int, optional
        Internal embedding dimension used throughout the transformer blocks.
        Each attention head operates on ``d_model // num_heads`` dimensions.
        Default: 64.
    ff_dim : int, optional
        Hidden dimension of the feed-forward sub-layers inside each
        transformer block.  Default: 128.

    Returns
    -------
    keras.Model
        Compiled Keras model.
        - Input shape : ``(batch_size, window_size, input_dim)``
        - Output shape: ``(batch_size, window_size, input_dim)``
        - Optimizer   : Adam (default learning rate)
        - Loss        : Mean Squared Error

    Raises
    ------
    ValueError
        If ``d_model`` is not divisible by ``num_heads``.
    """

    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
        )

    key_dim = d_model // num_heads
    inputs = Input(shape=(window_size, input_dim), name="input")

    # --- Encoder ---
    # Project each time step to the transformer embedding dimension
    x = layers.Dense(d_model, name="enc_embed")(inputs)

    # Self-attention sub-layer with residual + LayerNorm
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name="enc_mha"
    )(x, x)
    x = layers.LayerNormalization(name="enc_ln_1")(x + attn)

    # Feed-forward sub-layer with residual + LayerNorm
    ffn = layers.Dense(128, activation="relu", name="enc_ffn_intermediate")(x)
    ffn = layers.Dense(d_model, name="enc_ffn_out")(ffn)
    x = layers.LayerNormalization(name="enc_ln_2")(x + ffn)

    x = layers.Dense(32, activation="relu", name="enc_fnn_32")(x)
    # Pool across the time axis → (batch, d_model), then compress to bottleneck
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    encoded = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    # --- Decoder ---
    # Expand latent vector back to (batch, window_size, d_model)
    x = layers.Dense(window_size * 32, activation="relu", name="dec_dense")(encoded)
    x = layers.Reshape((window_size, 32), name="reshape")(x)

    x = layers.Dense(d_model, name="dec_upscale")(x)

    # Project back to the original feature dimension at each time step
    outputs = layers.TimeDistributed(
        layers.Dense(input_dim, activation="linear"), name="output"
    )(x)

    model = Model(inputs, outputs, name="transformer_autoencoder")
    model.compile(optimizer="adam", loss="mse")

    return model
