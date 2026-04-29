"""
prepare_device_splits.py
------------------------
Builds per-device train / val / test CSV splits for the federated learning
pipeline.

Each federated client (device) will be trained on its own normal-traffic data
and evaluated against a shared pool of attack traffic, simulating a realistic
anomaly-detection scenario where each device has a unique baseline but the
attack landscape is common.

Split strategy
--------------
For every device sub-directory found in data/normal_traffic/:
    - train.csv  : first 70 % of the device's rows (chronological order)
    - val.csv    : next  15 % of the device's rows
    - test.csv   : remaining 15 % of normal rows + ALL attack rows

Output
------
Splits are written to data/splits/<device_name>/ relative to the repository
root. The script must be run from the repository root so that the relative
data/ paths resolve correctly.

Usage:
    python src/utils/prepare_device_splits.py
"""

import pandas as pd
import os

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

NORMAL_DIR = os.path.join("data", "normal_traffic")
ATTACK_DIR = os.path.join("data", "attack_traffic")
SPLITS_DIR = os.path.join("data", "splits")


def load_csv(path: str) -> pd.DataFrame:

    """
    Load a CSV file into a DataFrame with mixed-type column inference disabled.

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        DataFrame containing all rows and columns from the file.
    """

    return pd.read_csv(path, low_memory = False)


def split_device(df: pd.DataFrame):

    """
    Split a device DataFrame into train, validation, and test subsets.

    Rows are kept in their original order (no shuffling) so the split respects
    the chronological nature of network traffic captures.

    Proportions are controlled by the module-level TRAIN_RATIO and VAL_RATIO
    constants. The test slice is the remaining rows after train and val.

    Args:
        df: Full DataFrame for a single device (normal traffic only).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def load_attack_data() -> pd.DataFrame:

    """
    Load and concatenate all attack-traffic CSVs from ATTACK_DIR into one DataFrame.

    All 14 attack-type CSV files are stacked vertically. The combined DataFrame
    is later appended to every device's test split so that each client is
    evaluated against the full attack corpus.

    Returns:
        Combined DataFrame of all attack rows, or an empty DataFrame if no CSV
        files are found in ATTACK_DIR.
    """

    frames = [
        load_csv(os.path.join(ATTACK_DIR, f))
        for f in sorted(os.listdir(ATTACK_DIR))
        if f.endswith(".csv")
    ]

    return pd.concat(frames, ignore_index = True) if frames else pd.DataFrame()


def main():

    """
    Entry point: generate and save per-device data splits.

    For each sub-directory (device) found in NORMAL_DIR:
        1. Load the device's CSV (expected at <NORMAL_DIR>/<device>/<device>.csv).
        2. Split it into train / val / normal-test subsets via split_device().
        3. Append all attack rows to the normal-test rows to form the final test set.
        4. Write train.csv, val.csv, and test.csv to SPLITS_DIR/<device>/.

    Prints a summary table with row counts and split percentages after processing
    all devices. Raises SystemExit if the required data directories are absent.
    """

    for d in (NORMAL_DIR, ATTACK_DIR):
        if not os.path.isdir(d):
            raise SystemExit(f"Directory not found: {d!r}\nRun this script from the repository root.")

    print("Loading attack traffic …", end = " ", flush = True)
    attack_df = load_attack_data()
    attack_files = sum(1 for f in os.listdir(ATTACK_DIR) if f.endswith(".csv"))
    print(f"{len(attack_df):,} rows loaded from {attack_files} files.\n")

    devices = sorted(
        d for d in os.listdir(NORMAL_DIR)
        if os.path.isdir(os.path.join(NORMAL_DIR, d))
    )
    print(f"Found {len(devices)} device(s): {', '.join(devices)}\n")

    summary_rows = []

    for device in devices:
        csv_path = os.path.join(NORMAL_DIR, device, f"{device}.csv")
        if not os.path.isfile(csv_path):
            print(f"[SKIP] {device}: expected CSV not found at {csv_path}")
            continue

        print(f"[{device}] Loading …", end = " ", flush = True)
        df = load_csv(csv_path)
        print(f"{len(df):,} rows loaded.")

        train, val, normal_test = split_device(df)

        # attack rows appended after normal test rows; file order is preserved within each
        test = pd.concat([normal_test, attack_df], ignore_index = True)

        out_dir = os.path.join(SPLITS_DIR, device)
        os.makedirs(out_dir, exist_ok = True)

        train.to_csv(os.path.join(out_dir, "train.csv"), index = False)
        val.to_csv(  os.path.join(out_dir, "val.csv"),   index = False)
        test.to_csv( os.path.join(out_dir, "test.csv"),  index = False)

        summary_rows.append({
            "device":      device,
            "train":       len(train),
            "val":         len(val),
            "test_normal": len(normal_test),
            "test_attack": len(attack_df),
            "train%":      f"{len(train) / len(df) * 100:.1f}",
            "val%":        f"{len(val)   / len(df) * 100:.1f}",
            "test%":       f"{len(normal_test) / len(df) * 100:.1f}",
        })
        print(f"         → train={len(train):,}  val={len(val):,}  "
              f"test={len(test):,} (normal={len(normal_test):,} + attack={len(attack_df):,})  "
              f"saved to {out_dir}/\n")

    print("=" * 84)
    print(f"{'Device':<30} {'Train':>8} {'Val':>7} {'Test(N)':>9} {'Test(A)':>9}  Train/Val/Test%")
    print("-" * 84)
    for r in summary_rows:
        print(f"{r['device']:<30} {r['train']:>8,} {r['val']:>7,} "
              f"{r['test_normal']:>9,} {r['test_attack']:>9,}  "
              f"{r['train%']}/{r['val%']}/{r['test%']}")
    print("=" * 84)
    print("Done. All splits written to data/splits/<device>/")


if __name__ == "__main__":
    main()
