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

    count = 0

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".pcap"):

                os.remove(os.path.join(dirpath, fname))
                count += 1
    return count


def find_parent_of_file(root: str, target: str) -> str | None:
    
    for dirpath, _, filenames in os.walk(root):
        if target in filenames:
            
            return dirpath
    return None


def print_tree(root: str, indent: int = 0, max_depth: int = 3) -> None:
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

    print(f"DOWNLOADING DATASET: {DATASET_SLUG} ...\n")
    dl_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"CACHED AT: {dl_path}\n")

    n_pcap = delete_pcap(dl_path)
    print(f"DELETED {n_pcap} .pcap FILES(S) FROM CACHE\n")

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

    print(f"ATTACK SOURCE: {attack_src}")
    print(f"NORMAL SOURCE: {normal_src}\n")

    os.makedirs(DATA_DIR, exist_ok = True)

    for src, dest, label in [
        (attack_src, ATTACK_DEST, "attack_traffic"),
        (normal_src, NORMAL_DEST, "normal_traffic"),
    ]:
        if os.path.exists(dest):
            print(f"REMOVING EXISTING {dest} ...")
            shutil.rmtree(dest)

        print(f"COPYING {label} ...")
        shutil.copytree(src, dest)
        csv_count = sum(
            1 for _, _, files in os.walk(dest) for f in files if f.endswith(".csv")
        )
        print(f"  -> {dest}  ({csv_count} CSV FILES)\n")


if __name__ == "__main__":
    main()
