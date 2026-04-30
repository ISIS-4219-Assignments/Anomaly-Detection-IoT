"""
models/
-------
Keras autoencoder definitions for the IoT anomaly detection simulation.

Available architectures
-----------------------
- vanilla  : Dense autoencoder. Input shape: (batch, n_features).
- lstm     : LSTM sequence autoencoder. Input shape: (batch, window, n_features).
- conv1d   : 1-D convolutional autoencoder. Input shape: (batch, window, n_features).

Entry point
-----------
Always use :func:`factory.build_model` to instantiate a model — never import
the architecture files directly from outside this package.  This keeps
``main.py`` and ``device.py`` decoupled from specific architecture choices.
"""
