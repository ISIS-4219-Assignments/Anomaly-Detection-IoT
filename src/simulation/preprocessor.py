"""
preprocessor.py
---------------

Stateful preprocessing pipeline for the IoT dataset used in the
federated learning simulation.

Typical usage per device
------------------------
::

    import pandas as pd
    from preprocessor import IoTPreprocessor, build_global_categories

    # --- Run once before any device trains ---
    all_train_paths = [
        "data/splits/device1/train.csv",
        "data/splits/device2/train.csv",
        ...
    ]
    known_categories = build_global_categories(all_train_paths)

    # --- Per device ---
    train_df = pd.read_csv("data/splits/<device>/train.csv")
    val_df   = pd.read_csv("data/splits/<device>/val.csv")
    test_df  = pd.read_csv("data/splits/<device>/test.csv")

    pre = IoTPreprocessor(known_categories=known_categories)

    X_train, y_train = pre.fit_transform(train_df)
    X_val,   y_val   = pre.transform(val_df)
    X_test,  y_test  = pre.transform(test_df)

Passing ``known_categories`` guarantees that every device produces
one-hot columns for **all** possible values seen across the federation,
not just those present in its own data.  This keeps the feature space
— and therefore the model architecture — identical across all devices,
which is a hard requirement for Federated Learning.

The scaler is fitted exclusively on training data and reused for val/test,
which is the correct approach in both standard ML and federated learning.
"""


from sklearn.preprocessing import StandardScaler
import pandas as pd


"""Columns removed before any modelling step.

These fields are either identifiers (IP addresses, timestamps), raw payloads
that carry no numeric signal, or port numbers that would leak label
information in a real deployment.  Any column listed here that is absent from
a given CSV is silently skipped.
"""


DROP_COLS = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
    "http.request.uri.query", "tcp.options", "tcp.payload",
    "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
    "tcp.checksum"
]


"""Columns stored as hex strings (e.g. '0x00000010') that must be parsed to
integers before the generic numeric coercion step.

These are protocol flag/bitmask fields whose integer values carry real signal
(e.g. TCP SYN/ACK/FIN bits).  They cannot be cast with pd.to_numeric directly
because that function does not recognise the '0x' prefix.
"""


HEX_COLS = ["tcp.flags", "mqtt.conflags", "mqtt.hdrflags"]


"""Label columns returned separately as *y* by fit_transform / transform.

``Attack_label`` is a binary flag (0 = normal, 1 = attack).
``Attack_type``  is a multiclass string identifying the specific attack
category (e.g. ``DDoS_HTTP_Flood``, ``SQL_injection``).
"""


TARGET_COLS = ["Attack_label", "Attack_type"]


"""String columns that are one-hot encoded before scaling.

Encoded column names follow the pattern ``<original_col>-<category>`` to
match the convention used in the dataset's original preprocessing guide.
Columns absent from a particular device's CSV are silently skipped.
"""


CATEGORICAL_COLS = [
    "http.request.method",
    "http.referer",
    "http.request.version",
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic",
]


def build_global_categories(train_paths: list[str]) -> dict[str, list]:

    """Scan all training CSVs and return every unique value per categorical column.

    This must be called once — before any device creates an
    :class:`IoTPreprocessor` — so that every device encodes categoricals
    against the same global vocabulary.  Passing the returned dict to
    ``IoTPreprocessor(known_categories=...)`` guarantees a uniform feature
    space across the entire federation.

    Only the columns listed in :data:`CATEGORICAL_COLS` are read from each
    file, so the function is fast even for large CSVs.

    Parameters
    ----------
    train_paths : list[str]
        Paths to every device's training CSV.

    Returns
    -------
    dict[str, list]
        ``{column_name: sorted_list_of_unique_values}`` for every
        categorical column that appears in at least one training file.
    """
    
    categories: dict[str, set] = {col: set() for col in CATEGORICAL_COLS}
    for path in train_paths:
        df = pd.read_csv(path, usecols = lambda c: c in CATEGORICAL_COLS)
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                categories[col].update(df[col].dropna().unique())
    return {col: sorted(vals) for col, vals in categories.items() if vals}


def _parse_hex_value(v):

    """Convert a single value from a HEX_COLS column to a numeric type.

    Handles three formats found across train/val/test splits:

    - ``'0x00000010'`` — hex string → parsed as integer via ``int(v, 0)``.
    - ``'0.0'`` / ``'16.0'`` — float string → parsed via ``float(v)``.
    - Already numeric (int or float) — returned unchanged.

    Non-parseable strings become ``float('nan')`` and are later removed by
    the ``dropna`` step in :meth:`IoTPreprocessor._clean`.
    """

    if not isinstance(v, str):
        return v
    try:
        return int(v, 0)   # '0x...' or plain integer string
    except ValueError:
        try:
            return float(v)   # '0.0', '16.0', etc.
        except ValueError:
            return float("nan")


class IoTPreprocessor:

    """Stateful preprocessor for the IoT dataset.

    The pipeline applies, in order:

    1. **Column dropping** — removes identifiers, payloads, and port numbers
       listed in :data:`DROP_COLS`.
    2. **Row cleaning** — drops rows with any null value and removes exact
       duplicates.
    3. **One-hot encoding** — expands the string columns in
       :data:`CATEGORICAL_COLS` into binary indicator columns.
    4. **Standard scaling** — zero-centres and unit-variance-scales all
       feature columns using a :class:`~sklearn.preprocessing.StandardScaler`
       that is *fitted only on training data*.

    The preprocessor is stateful: after :meth:`fit_transform` is called on the
    training split, the fitted scaler and the ordered list of feature columns
    are stored on the instance.  Subsequent calls to :meth:`transform` reuse
    both, which guarantees that val and test splits are processed identically
    to training data.

    Federated Learning — uniform feature space
    ------------------------------------------
    In a federation each device sees only a subset of the data, so a naive
    ``pd.get_dummies`` call produces a different set of indicator columns per
    device.  This breaks model aggregation because the weight tensors have
    different shapes.

    Pass the dict returned by :func:`build_global_categories` as
    ``known_categories`` to fix this.  :meth:`_encode` then forces every
    categorical column to produce exactly the same indicator columns on every
    device — categories absent from a device's data become all-zero columns
    instead of missing entirely.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted after the first call to :meth:`fit_transform`.
    feature_columns : list[str] or None
        Ordered list of feature column names produced by the training split.
        ``None`` until :meth:`fit_transform` is called.
    known_categories : dict[str, list] or None
        Global vocabulary ``{col: [values]}`` supplied at construction time.
        When set, the feature space is fixed to these categories regardless of
        what each device's local data contains.
    """

    def __init__(self, known_categories: dict[str, list] | None = None):
        self.scaler = StandardScaler()
        self.feature_columns: list[str] | None = None
        self.known_categories = known_categories


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        """Clean, encode, and scale the training split.

        This method fits the internal :class:`~sklearn.preprocessing.StandardScaler`
        on the cleaned, encoded features and records the resulting column
        schema.  It must be called before :meth:`transform`.

        Parameters
        ----------
        df : pd.DataFrame
            Raw training split as loaded directly from a CSV file.

        Returns
        -------
        X : pd.DataFrame
            Scaled feature matrix with shape ``(n_samples, n_features)``.
        y : pd.DataFrame
            Label columns (``Attack_label`` and/or ``Attack_type``) with shape
            ``(n_samples, n_targets)``.  Only columns present in *df* are
            included.
        """

        df = self._clean(df)
        df = self._encode(df)
        X, y = self._split_xy(df)
        self.feature_columns = X.columns.tolist()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns = self.feature_columns,
            index = X.index,
        )
        return X_scaled, y


    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        """Clean, encode, and scale a val or test split using the fitted scaler.

        Applies the same cleaning and encoding steps as :meth:`fit_transform`
        but uses the already-fitted scaler parameters.  The feature matrix is
        reindexed to match the column schema learned during training, so the
        output is always compatible with a model trained on ``fit_transform``
        output.

        Parameters
        ----------
        df : pd.DataFrame
            Raw val or test split as loaded directly from a CSV file.

        Returns
        -------
        X : pd.DataFrame
            Scaled feature matrix aligned to the training column schema.
        y : pd.DataFrame
            Label columns present in *df*.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit_transform` has been called on this
            instance.
        """

        if self.feature_columns is None:
            raise RuntimeError("Call fit_transform() on training data before transform().")
        df = self._clean(df)
        df = self._encode(df)
        X, y = self._split_xy(df)
        # Align to training schema: add missing columns as 0, drop extras
        X = X.reindex(columns = self.feature_columns, fill_value = 0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns = self.feature_columns,
            index = X.index,
        )
        return X_scaled, y


    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------


    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:

        """Drop unwanted columns, null rows, and duplicate rows.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (modified via copies, not in-place).

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with a fresh integer index.
        """

        existing = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(columns = existing)

        # Parse hex-string flag columns (e.g. '0x00000010') to integers.
        # pd.to_numeric does not recognise the '0x' prefix, so these columns
        # must be handled before the generic coercion step below.
        for col in HEX_COLS:
            if col in df.columns:
                df[col] = df[col].apply(_parse_hex_value)

        # Coerce remaining feature columns to numeric, turning any
        # non-parseable residual values into NaN for the dropna step.
        skip = set(TARGET_COLS) | set(CATEGORICAL_COLS)
        numeric_cols = [c for c in df.columns if c not in skip]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors = "coerce")

        df = df.dropna()
        df = df.drop_duplicates()
        return df.reset_index(drop = True)


    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:

        """One-hot encode the categorical columns listed in :data:`CATEGORICAL_COLS`.

        Each categorical column ``col`` is replaced by binary indicator columns
        named ``col-<category>``.

        When ``self.known_categories`` is set (the recommended FL path), the
        dummy columns for every categorical are reindexed to exactly the global
        vocabulary supplied at construction time.  This means:

        * Categories present in the global vocab but absent from this device's
          data become all-zero columns (not dropped).
        * Categories in this device's data that are not in the global vocab are
          silently dropped (they would produce unseen model inputs).
        * Categorical columns entirely absent from *df* are still represented as
          all-zero dummy columns in the output.

        When ``self.known_categories`` is ``None`` the method falls back to the
        local-only behaviour: columns absent from *df* are skipped and only the
        categories observed locally are encoded.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame after :meth:`_clean`.

        Returns
        -------
        pd.DataFrame
            DataFrame with categorical columns replaced by indicator columns.
        """

        for col in CATEGORICAL_COLS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix = col, prefix_sep = "-")
                df = df.drop(columns = [col])
            else:
                # Column absent from this device; start with an empty frame
                # so the reindex below can still add the expected zero columns.
                dummies = pd.DataFrame(index = df.index)

            if self.known_categories and col in self.known_categories:
                expected = [f"{col}-{cat}" for cat in self.known_categories[col]]
                dummies = dummies.reindex(columns = expected, fill_value = 0)

            if not dummies.empty:
                df = pd.concat([df, dummies], axis = 1)

        return df


    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        """Separate features from label columns.

        Parameters
        ----------
        df : pd.DataFrame
            Fully encoded DataFrame.

        Returns
        -------
        X : pd.DataFrame
            All columns except those in :data:`TARGET_COLS`.
        y : pd.DataFrame
            Only the target columns that are present in *df*.
        """

        target_present = [c for c in TARGET_COLS if c in df.columns]
        y = df[target_present]
        X = df.drop(columns = target_present)
        return X, y
