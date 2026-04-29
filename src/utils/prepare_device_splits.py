"""
prepare_device_splits.py
------------------------
Builds per-device train / val / test CSV splits for the federated learning
pipeline.

Split strategy
--------------
For every device sub-directory found in data/normal_traffic/:
    - train.csv  : first 70 % of device rows (capped at MAX_NORMAL_ROWS)
    - val.csv    : next  15 % of device rows
    - test.csv   : interleaved normal segments and assigned attack chunks

Attack assignment
-----------------
The 14 attack CSVs are distributed round-robin across devices (both sorted
alphabetically) so every attack appears in exactly one device's test set
and all attacks are covered.

Test set layout per device (example with 2 assigned attacks)
-------------------------------------------------------------
    [normal_seg_1]
    + [attack_1 rows sampled to budget]
    + [normal_seg_2]
    + [attack_2 rows sampled to budget]
    + [normal_seg_3]

The full 15 % normal-test slice is split evenly into (n_attacks + 1) segments
so that every attack is bracketed by normal traffic on both sides.
Total attack rows across all assigned attacks are capped at ATTACK_FACTOR *
loaded_rows; each attack's share is proportional to its original size.

Output
------
Splits are written to data/splits/<device_name>/ relative to the repository
root. Run this script from the repository root.

Usage:
    python src/utils/prepare_device_splits.py
"""

import pandas as pd
import os


TRAIN_RATIO         = 0.70
VAL_RATIO           = 0.15
CAP_FACTOR    = 0.2   # load only 20% of each device's rows
ATTACK_FACTOR = 0.1   # attack rows in test capped at 10% of loaded normal rows

NORMAL_DIR = os.path.join("data", "normal_traffic")
ATTACK_DIR = os.path.join("data", "attack_traffic")
SPLITS_DIR = os.path.join("data", "splits")

EXCLUDED_DEVICES = {"Heart_Rate", "Modbus"}


def load_csv(path: str) -> pd.DataFrame:

    """
    Read a CSV file into a DataFrame without type inference on mixed columns.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw contents of the file with all columns retained.
    """

    return pd.read_csv(path, low_memory = False)


def split_device(df: pd.DataFrame):

    """
    Partition a device's DataFrame into train, validation, and test slices.

    The split is purely positional (no shuffling) to preserve temporal order
    of network captures.  Proportions are governed by TRAIN_RATIO and
    VAL_RATIO; the remainder becomes the normal-traffic test slice.

    Parameters
    ----------
    df : pd.DataFrame
        Full normal-traffic DataFrame for one device.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, normal_test) — three non-overlapping slices of ``df``.
    """

    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def load_attack_chunks() -> dict:

    """
    Load every attack CSV from ATTACK_DIR into memory.

    Files are read in alphabetical order so the resulting dictionary is
    deterministically ordered, which matters for the round-robin attack
    assignment that follows.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from attack name (CSV filename without extension) to its
        full DataFrame.
    """

    chunks = {}
    for f in sorted(os.listdir(ATTACK_DIR)):
        if f.endswith(".csv"):
            chunks[f[:-4]] = load_csv(os.path.join(ATTACK_DIR, f))
    return chunks


def assign_attacks(attack_names: list, devices: list) -> dict:

    """
    Distribute attacks across devices using a round-robin strategy.

    Both ``attack_names`` and ``devices`` must already be sorted so that the
    assignment is reproducible across runs.  The modulo index ensures every
    attack lands on exactly one device and no attack is skipped even when the
    number of attacks is not a multiple of the number of devices.

    Parameters
    ----------
    attack_names : list[str]
        Ordered list of attack identifiers (keys from load_attack_chunks).
    devices : list[str]
        Ordered list of device directory names found in NORMAL_DIR.

    Returns
    -------
    dict[str, list[str]]
        Mapping from device name to the list of attack names assigned to it.
        Devices with no attacks receive an empty list.
    """

    assignment = {d: [] for d in devices}
    for i, attack in enumerate(attack_names):
        # Cycle through devices so attacks spread evenly regardless of count
        assignment[devices[i % len(devices)]].append(attack)
    return assignment


def build_test_set(normal_test_df: pd.DataFrame, attack_chunks: list, attack_budget: int) -> pd.DataFrame:

    """
    Construct a test set by interleaving normal segments with sampled attack chunks.

    Pattern: normal_seg_1 | attack_1 | normal_seg_2 | attack_2 | ... | normal_seg_n+1

    All normal-test rows are used, split evenly into (n_attacks + 1) segments so
    every attack is bracketed by normal traffic on both sides.  Attack chunks are
    down-sampled proportionally if their combined row count exceeds attack_budget.

    Parameters
    ----------
    normal_test_df : pd.DataFrame
        Normal-traffic rows reserved for the test split of a given device.
    attack_chunks : list[pd.DataFrame]
        Ordered list of attack DataFrames assigned to this device.
    attack_budget : int
        Maximum total attack rows allowed (ATTACK_FACTOR * loaded normal rows).

    Returns
    -------
    pd.DataFrame
        Concatenated test set with a fresh integer index.
    """

    if not attack_chunks:
        return normal_test_df.reset_index(drop = True)

    # Down-sample each attack proportionally so combined rows <= attack_budget
    total_attack_rows = sum(len(c) for c in attack_chunks)
    if total_attack_rows > attack_budget:
        scale   = attack_budget / total_attack_rows
        sampled = [c.iloc[:max(1, int(len(c) * scale))] for c in attack_chunks]
    else:
        sampled = attack_chunks

    # Split normal pool evenly into (n_attacks + 1) segments
    pool     = normal_test_df.reset_index(drop = True)
    n        = len(sampled)
    seg_size = len(pool) // (n + 1)
    pieces   = []
    offset   = 0

    for chunk in sampled:
        pieces.append(pool.iloc[offset : offset + seg_size])
        pieces.append(chunk)
        offset += seg_size

    pieces.append(pool.iloc[offset:])  # trailing normal segment

    return pd.concat(pieces, ignore_index = True)


def main():

    """
    Orchestrate the full split-generation pipeline.

    Steps
    -----
    1. Validate that the required source directories exist.
    2. Load all attack CSVs and assign them round-robin to devices.
    3. For each device:
       a. Load and optionally cap its normal-traffic CSV.
       b. Compute train / val / normal-test slices.
       c. Build the interleaved test set.
       d. Write train.csv, val.csv, test.csv to data/splits/<device>/.
    4. Print a summary table of row counts and attack assignments.
    """
    
    for d in (NORMAL_DIR, ATTACK_DIR):
        if not os.path.isdir(d):
            raise SystemExit(
                f"Directory not found: {d!r}\nRun this script from the repository root."
            )

    print("Loading attack traffic …", end = " ", flush = True)
    attack_chunks = load_attack_chunks()
    attack_names  = list(attack_chunks.keys())
    total_attack  = sum(len(v) for v in attack_chunks.values())
    print(f"{total_attack:,} rows across {len(attack_chunks)} files.\n")

    # Collect device names from sub-directories only (skip loose files and excluded devices)
    devices = sorted(
        d for d in os.listdir(NORMAL_DIR)
        if os.path.isdir(os.path.join(NORMAL_DIR, d)) and d not in EXCLUDED_DEVICES
    )
    print(f"Found {len(devices)} device(s): {', '.join(devices)}\n")

    assignment = assign_attacks(attack_names, devices)

    summary_rows = []

    for device in devices:
        # Expected convention: each device folder contains a same-named CSV
        csv_path = os.path.join(NORMAL_DIR, device, f"{device}.csv")
        if not os.path.isfile(csv_path):
            print(f"[SKIP] {device}: expected CSV not found at {csv_path}")
            continue

        print(f"[{device}] Loading …", end = " ", flush = True)

        df = load_csv(csv_path)
        print(f"Rows identified (100%): {len(df)} ")

        cap_limit = int(len(df) * CAP_FACTOR)
        df = df.iloc[:cap_limit]
        print(f"Rows loaded (20%): {len(df)}")

        train, val, normal_test = split_device(df)

        assigned      = assignment[device]
        chunks        = [attack_chunks[a] for a in assigned]
        attack_budget = int(len(df) * ATTACK_FACTOR)
        test          = build_test_set(normal_test, chunks, attack_budget)

        out_dir = os.path.join(SPLITS_DIR, device)
        os.makedirs(out_dir, exist_ok = True)

        train.to_csv(os.path.join(out_dir, "train.csv"), index = False)
        val.to_csv(  os.path.join(out_dir, "val.csv"),   index = False)
        test.to_csv( os.path.join(out_dir, "test.csv"),  index = False)

        n_attack     = len(test) - len(normal_test)
        attack_label = ", ".join(assigned) if assigned else "—"
        print(f"         → train={len(train):,}  val={len(val):,}  "
              f"test={len(test):,} (normal={len(normal_test):,} + attack={n_attack:,})\n"
              f"           attacks: [{attack_label}]\n"
              f"           saved to {out_dir}/\n")

        summary_rows.append({
            "device":      device,
            "train":       len(train),
            "val":         len(val),
            "test_normal": len(normal_test),
            "test_attack": n_attack,
            "test_total":  len(test),
            "attacks":     attack_label,
        })

    col = 104
    print("=" * col)
    print(f"{'Device':<30} {'Train':>8} {'Val':>7} {'Test(Normal)':>9} {'Test(Attack)':>9} {'Test(Total)':>7}  Attacks")
    print("-" * col)
    for r in summary_rows:
        print(f"{r['device']:<30} {r['train']:>8,} {r['val']:>7,} "
              f"{r['test_normal']:>9,} {r['test_attack']:>9,} {r['test_total']:>7,}  {r['attacks']}")
    print("=" * col)
    print("Done. All splits written to data/splits/<device>/")


if __name__ == "__main__":
    main()
