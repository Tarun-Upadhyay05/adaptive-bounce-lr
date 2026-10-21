"""Download the official CIFAR-10 Python archive and check its MD5.

The torchvision default host is the University of Toronto. If that host is
slow, pass ``--url`` with a mirror of the same archive. Training still reads
the extracted official batches through torchvision, and the checksum below is
the one published with that archive.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

OFFICIAL_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
OFFICIAL_MD5 = "c58f30108f718f92721af3b95e74349a"
FILENAME = "cifar-10-python.tar.gz"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract official CIFAR-10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--url", default=OFFICIAL_URL)
    args = parser.parse_args()
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / FILENAME
    extracted = root / "cifar-10-batches-py" / "data_batch_1"
    if extracted.exists():
        print(f"already extracted: {extracted}")
        return
    if not archive.exists() or md5(archive) != OFFICIAL_MD5:
        print(f"downloading {args.url}")
        urllib.request.urlretrieve(args.url, archive)
    checksum = md5(archive)
    if checksum != OFFICIAL_MD5:
        raise SystemExit(f"checksum {checksum} does not match official {OFFICIAL_MD5}")
    print("checksum matches the official CIFAR-10 Python archive")
    with tarfile.open(archive, "r:gz") as handle:
        if sys.version_info >= (3, 12):
            handle.extractall(root, filter="data")
        else:
            handle.extractall(root)
    print(f"extracted to {root / 'cifar-10-batches-py'}")


if __name__ == "__main__":
    main()
