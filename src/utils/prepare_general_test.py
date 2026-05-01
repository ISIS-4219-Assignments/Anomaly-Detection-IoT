"""
prepare_general_test.py
-----------------------

Builds a single general test CSV that covers ALL 14 attack types and normal
traffic pooled from ALL devices.  This file is used to evaluate the
generalisation and shared knowledge of the final global federated model.

Motivation
----------
Per-device test sets each carry only a subset of attacks (round-robin
assignment in prepare_device_splits.py).  A model that scores well on every
device test set might still miss attack types it has never encountered locally.
This general test set puts all 14 attack types in one file so the global model
can be evaluated end-to-end.

Split strategy
--------------
- Normal rows  : test slices (last 15 % after the 20 % row cap) from every
  device's normal-traffic CSV, concatenated across all devices.
- Attack rows  : all 14 attack CSVs, sampled proportionally when their
  combined row count exceeds  ATTACK_FACTOR × total_normal_rows.

Layout
------
    [normal_seg_1] [attack_1] [normal_seg_2] [attack_2] … [attack_14] [normal_seg_15]

Each attack is bracketed by normal traffic on both sides, mirroring the
structure used in per-device test sets.

Output
------
    data/splits/general_test.csv   (run from repository root)

Usage
-----
    python src/utils/prepare_general_test.py
"""

import pandas as pd
import os


TRAIN_RATIO   = 0.70   # must match prepare_device_splits.py
VAL_RATIO     = 0.15   # must match prepare_device_splits.py
CAP_FACTOR    = 0.20   # load only 20 % of each device's rows
ATTACK_FACTOR = 0.10   # attack rows capped at 10 % of total normal rows

NORMAL_DIR  = os.path.join("data", "normal_traffic")
ATTACK_DIR  = os.path.join("data", "attack_traffic")
SPLITS_DIR  = os.path.join("data", "splits")
OUTPUT_PATH = os.path.join(SPLITS_DIR, "general_test.csv")

EXCLUDED_DEVICES = {"Heart_Rate", "Modbus"}


def load_csv(path: str) -> pd.DataFrame:

    """Read a CSV without type inference on mixed-type columns."""

    return pd.read_csv(path, low_memory=False)


def build_test_set(
    normal_df: pd.DataFrame,
    attack_chunks: list[pd.DataFrame],
    attack_budget: int,
) -> pd.DataFrame:

    """Interleave normal segments with (optionally down-sampled) attack chunks.

    Identical logic to prepare_device_splits.build_test_set so both files
    produce structurally equivalent test sets.

    Pattern: normal_seg_1 | attack_1 | normal_seg_2 | … | attack_n | normal_seg_n+1
    """

    if not attack_chunks:
        return normal_df.reset_index(drop=True)

    total_attack_rows = sum(len(c) for c in attack_chunks)
    if total_attack_rows > attack_budget:
        scale   = attack_budget / total_attack_rows
        sampled = [c.iloc[: max(1, int(len(c) * scale))] for c in attack_chunks]
    else:
        sampled = attack_chunks

    pool     = normal_df.reset_index(drop=True)
    n        = len(sampled)
    seg_size = len(pool) // (n + 1)
    pieces   = []
    offset   = 0

    for chunk in sampled:
        pieces.append(pool.iloc[offset : offset + seg_size])
        pieces.append(chunk)
        offset += seg_size

    pieces.append(pool.iloc[offset:])   # trailing normal segment
    return pd.concat(pieces, ignore_index=True)


def main():

    """Build and save the general test CSV.

    Steps
    -----
    1. Load all 14 attack CSVs from ATTACK_DIR.
    2. Collect the normal test slice from every eligible device.
    3. Concatenate the normal slices into a single pool.
    4. Interleave the pool with all attack chunks (budget-capped).
    5. Write the result to OUTPUT_PATH.
    """

    for d in (NORMAL_DIR, ATTACK_DIR):
        if not os.path.isdir(d):
            raise SystemExit(
                f"Directory not found: {d!r}\nRun this script from the repository root."
            )

    # --- 1. Load all attack CSVs ---
    print("Loading attack traffic …", end=" ", flush=True)
    attack_chunks: dict[str, pd.DataFrame] = {}
    for f in sorted(os.listdir(ATTACK_DIR)):
        if f.endswith(".csv"):
            attack_chunks[f[:-4]] = load_csv(os.path.join(ATTACK_DIR, f))
    total_attack = sum(len(v) for v in attack_chunks.values())
    print(f"{total_attack:,} rows across {len(attack_chunks)} attack type(s).\n")

    # --- 2. Collect normal test slices from all devices ---
    devices = sorted(
        d for d in os.listdir(NORMAL_DIR)
        if os.path.isdir(os.path.join(NORMAL_DIR, d)) and d not in EXCLUDED_DEVICES
    )
    print(f"Collecting normal test slices from {len(devices)} device(s):")

    normal_pieces: list[pd.DataFrame] = []
    for device in devices:
        csv_path = os.path.join(NORMAL_DIR, device, f"{device}.csv")
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {device}: CSV not found at {csv_path}")
            continue

        df  = load_csv(csv_path)
        cap = int(len(df) * CAP_FACTOR)
        df  = df.iloc[:cap]

        val_end    = int(len(df) * (TRAIN_RATIO + VAL_RATIO))
        test_slice = df.iloc[val_end:].reset_index(drop=True)
        normal_pieces.append(test_slice)
        print(f"  [{device:<26}]  capped={len(df):>8,}  →  test slice={len(test_slice):>7,}")

    if not normal_pieces:
        raise SystemExit("No device data found — cannot build general test set.")

    # --- 3. Pool normal rows ---
    normal_df = pd.concat(normal_pieces, ignore_index=True)
    print(f"\nTotal normal rows pooled across all devices: {len(normal_df):,}")

    # --- 4. Build interleaved test set ---
    attack_list   = list(attack_chunks.values())
    attack_names  = list(attack_chunks.keys())
    attack_budget = int(len(normal_df) * ATTACK_FACTOR)

    print(f"Attack budget: {attack_budget:,} rows  ({ATTACK_FACTOR:.0%} × normal rows)")
    print(f"Attack types : {len(attack_names)}")

    general_test = build_test_set(normal_df, attack_list, attack_budget)

    # --- 5. Save ---
    os.makedirs(SPLITS_DIR, exist_ok=True)
    general_test.to_csv(OUTPUT_PATH, index=False)

    if "Attack_label" in general_test.columns:
        n_normal = int((general_test["Attack_label"] == 0).sum())
        n_attack = int((general_test["Attack_label"] == 1).sum())
    else:
        n_normal = n_attack = "n/a"

    col = 72
    print("\n" + "=" * col)
    print(f"General test set written to: {OUTPUT_PATH}")
    print(f"  Total rows  : {len(general_test):,}")
    print(f"  Normal rows : {n_normal:,}" if isinstance(n_normal, int) else f"  Normal rows : {n_normal}")
    print(f"  Attack rows : {n_attack:,}" if isinstance(n_attack, int) else f"  Attack rows : {n_attack}")
    print(f"  Attack types ({len(attack_names)}): {', '.join(attack_names)}")
    print("=" * col)


if __name__ == "__main__":
    main()
