"""
preprocessor.py
---------------
Stateful preprocessing pipeline for the IoT dataset used in the
federated learning simulation.

Typical usage per device
------------------------
::

    import pandas as pd
    from preprocessor import IoTPreprocessor

    train_df = pd.read_csv("data/splits/<device>/train.csv")
    val_df   = pd.read_csv("data/splits/<device>/val.csv")
    test_df  = pd.read_csv("data/splits/<device>/test.csv")

    pre = IoTPreprocessor()

    X_train, y_train = pre.fit_transform(train_df)
    X_val,   y_val   = pre.transform(val_df)
    X_test,  y_test  = pre.transform(test_df)

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
]


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

    Column alignment
    ----------------
    Because each device in the federation captures a different subset of
    network protocols, the set of one-hot columns produced by encoding can
    differ between splits.  :meth:`transform` calls
    ``DataFrame.reindex(columns=self.feature_columns, fill_value=0)`` so that
    the output always has exactly the same columns (in the same order) as the
    training split — new categories seen at inference time are dropped, and
    categories absent from a split are filled with zero.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted after the first call to :meth:`fit_transform`.
    feature_columns : list[str] or None
        Ordered list of feature column names produced by the training split.
        ``None`` until :meth:`fit_transform` is called.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns: list[str] | None = None

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
            columns=self.feature_columns,
            index=X.index,
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
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_columns,
            index=X.index,
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
        df = df.drop(columns=existing)
        df = df.dropna()
        df = df.drop_duplicates()
        return df.reset_index(drop=True)

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:

        """One-hot encode the categorical columns listed in :data:`CATEGORICAL_COLS`.

        Each categorical column ``col`` is replaced by binary indicator columns
        named ``col-<category>``.  Columns absent from *df* are silently
        skipped so the same preprocessor works across devices with different
        protocol mixes.

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
            if col not in df.columns:
                continue
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep="-")
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
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
        X = df.drop(columns=target_present)
        return X, y
