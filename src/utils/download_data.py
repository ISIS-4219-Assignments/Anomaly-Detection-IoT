"""
download_data.py
----------------
Downloads the EdgeIIoTSet dataset from Kaggle via kagglehub and organises it
into the project's data/ directory.

Usage (run from any directory):
    python src/utils/download_data.py

Requires the KAGGLE_USERNAME and KAGGLE_KEY environment variables (or a
~/.kaggle/kaggle.json credentials file) to be present so kagglehub can
authenticate.

After a successful run the project data layout will be:
    data/
        attack_traffic/   <- CSV files for 14 attack categories
        normal_traffic/   <- CSV files for normal IoT sensor readings
"""

import kagglehub
import shutil
import sys
import os


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
ATTACK_DEST  = os.path.join(DATA_DIR, "attack_traffic")
NORMAL_DEST  = os.path.join(DATA_DIR, "normal_traffic")

ATTACK_SENTINEL = "Backdoor_attack.csv"
NORMAL_SENTINEL = "Distance.csv"

DATASET_SLUG = "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"


def delete_pcap(root: str) -> int:

    """
    Recursively delete every .pcap file under *root*.

    .pcap files are large raw packet captures that are not needed for the CSV-
    based analysis. Removing them saves significant disk space in the kagglehub
    cache directory.

    Args:
        root: Top-level directory to search.

    Returns:
        Number of .pcap files deleted.
    """

    count = 0

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".pcap"):

                os.remove(os.path.join(dirpath, fname))
                count += 1

    return count


def find_parent_of_file(root: str, target: str) -> str | None:
    
    """
    Walk *root* and return the first directory that directly contains *target*.

    Used to locate the attack-traffic and normal-traffic folders inside the
    kagglehub cache, whose exact path can vary between dataset versions.

    Args:
        root:   Directory to search recursively.
        target: Filename (not a path) to look for.

    Returns:
        Absolute path of the directory containing *target*, or None if not found.
    """

    for dirpath, _, filenames in os.walk(root):
        if target in filenames:
            
            return dirpath
    return None


def print_tree(root: str, indent: int = 0, max_depth: int = 3) -> None:
    
    """
    Print an indented directory tree rooted at *root* up to *max_depth* levels.

    Intended for diagnostic output when a sentinel file cannot be found so the
    user can inspect what was actually downloaded.

    Args:
        root:      Directory to print.
        indent:    Current indentation level (used in recursive calls).
        max_depth: Maximum number of levels to descend before stopping.
    """

    if indent > max_depth:
        return
    prefix = "  " * indent

    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return
    for entry in entries:
        full = os.path.join(root, entry)
        marker = "/" if os.path.isdir(full) else ""
        print(f"{prefix}{entry}{marker}")
        if os.path.isdir(full):
            print_tree(full, indent + 1, max_depth)


def main() -> None:

    """
    Orchestrate the full dataset download-and-organise pipeline.

    Steps:
        1. Download (or use the cached version of) the EdgeIIoTSet dataset via
           kagglehub.
        2. Remove all .pcap files from the cache to free disk space.
        3. Locate the attack-traffic and normal-traffic source folders by
           searching for known sentinel filenames.
        4. Copy both folders into data/attack_traffic/ and data/normal_traffic/,
           replacing any existing copies.

    Exits with code 1 if either source folder cannot be found in the download.
    """
    
    print(f"Downloading dataset: {DATASET_SLUG} ...\n")
    dl_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Cached at: {dl_path}\n")

    n_pcap = delete_pcap(dl_path)
    print(f"Deleted {n_pcap} .pcap files(s) from cache\n")

    attack_src = find_parent_of_file(dl_path, ATTACK_SENTINEL)
    normal_file_path = None
    for dirpath, _, filenames in os.walk(dl_path):
        if NORMAL_SENTINEL in filenames:
            normal_file_path = dirpath
            break

    if normal_file_path:
        normal_src = os.path.dirname(normal_file_path)
        if not normal_src.startswith(dl_path):
            normal_src = normal_file_path
    else:
        normal_src = None

    if not attack_src:
        print(f"ERROR: Could not find the attack-traffic folder (looked for '{ATTACK_SENTINEL}').\nDownloaded tree:", file = sys.stderr)
        print_tree(dl_path)
        sys.exit(1)

    if not normal_src:
        print(f"ERROR: Could not find the normal-traffic folder (looked for '{NORMAL_SENTINEL}').\nDownloaded tree:", file = sys.stderr)
        print_tree(dl_path)
        sys.exit(1)

    print(f"Attack source: {attack_src}")
    print(f"Normal source: {normal_src}\n")

    os.makedirs(DATA_DIR, exist_ok = True)

    for src, dest, label in [
        (attack_src, ATTACK_DEST, "attack_traffic"),
        (normal_src, NORMAL_DEST, "normal_traffic"),
    ]:
        if os.path.exists(dest):
            print(f"Removing existing {dest} ...")
            shutil.rmtree(dest)

        print(f"Copying {label} ...")
        shutil.copytree(src, dest)
        csv_count = sum(
            1 for _, _, files in os.walk(dest) for f in files if f.endswith(".csv")
        )
        print(f"  -> {dest}  ({csv_count} csv files)\n")


if __name__ == "__main__":
    main()
