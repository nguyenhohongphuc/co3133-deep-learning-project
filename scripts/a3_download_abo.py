"""Download official ABO metadata and the image index.

Run from the repository root: python scripts/a3_download_abo.py
Byte ranges allow interrupted downloads to resume without restarting the archive.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


BASE_URL = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"
FILES = (
    "archives/abo-listings.tar",
    "images/metadata/images.csv.gz",
    "README.md",
    "LICENSE-CC-BY-4.0.txt",
)


def download(source: str, destination: Path, workers: int) -> dict:
    url = BASE_URL + source
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
        etag = response.headers["ETag"]
    parts = destination.with_name(destination.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    identity = {"url": url, "size": size, "etag": etag}
    identity_path = parts / "source.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError(f"Remote file changed. Inspect previous parts in {parts} before retrying.")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    chunk_size = 16 * 1024 * 1024
    ranges = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]

    def fetch(bounds: tuple[int, int]) -> Path:
        start, end = bounds
        path = parts / f"{start:012d}.part"
        expected = end - start + 1
        for attempt in range(6):
            completed = path.stat().st_size if path.exists() else 0
            if completed == expected:
                return path
            if completed > expected:
                raise RuntimeError(f"Oversized download part: {path}")
            headers = {"Range": f"bytes={start + completed}-{end}", "If-Match": etag}
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    actual_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or actual_range != f"bytes {start + completed}-{end}/{size}":
                        raise RuntimeError(f"Unexpected range response: {response.status}, {actual_range}")
                    with path.open("ab") as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                if path.stat().st_size != expected:
                    raise OSError("Incomplete range")
                return path
            except (OSError, TimeoutError, http.client.HTTPException) as error:
                if attempt == 5:
                    raise
                print(f"Retry {destination.name} at {start}: {error}", flush=True)
                time.sleep(min(2 ** attempt, 15))
        raise RuntimeError("Download failed")

    # A complete destination is checked against the source fingerprint from an earlier run.
    receipt_path = destination.with_name(destination.name + ".json")
    if destination.exists() and destination.stat().st_size == size and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if all(receipt.get(key) == value for key, value in identity.items()):
            checksum = hashlib.sha256()
            with destination.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    checksum.update(block)
            if checksum.hexdigest() == receipt["sha256"]:
                print(f"Verified existing {destination.name}", flush=True)
                for path in parts.glob("*.part"):
                    path.unlink()
                identity_path.unlink()
                parts.rmdir()
                return receipt

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for count, _ in enumerate(executor.map(fetch, ranges), 1):
            print(f"{destination.name}: {count}/{len(ranges)} ranges complete", flush=True)
    temporary = destination.with_name(destination.name + ".assembling")
    checksum = hashlib.sha256()
    with temporary.open("wb") as output:
        for start, _ in ranges:
            with (parts / f"{start:012d}.part").open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    checksum.update(block)
                    output.write(block)
    if temporary.stat().st_size != size:
        raise RuntimeError(f"Unexpected archive size: {temporary}")
    os.replace(temporary, destination)
    receipt = {**identity, "sha256": checksum.hexdigest()}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    for path in parts.glob("*.part"):
        path.unlink()
    identity_path.unlink()
    parts.rmdir()
    print(f"Saved {destination.name} ({size:,} bytes)", flush=True)
    return receipt


def download_images(relative_paths: list[str], root: Path, workers: int) -> None:
    """Fetch native 256px main images using persistent connections to the official S3 bucket."""
    ledger = root / "image_sources.jsonl"
    receipts = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                record = json.loads(line)
                receipts[record["path"]] = record
            except json.JSONDecodeError:
                continue
    local = threading.local()

    def fetch(relative: str) -> dict:
        destination = root / relative
        if not destination.resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"Invalid source path: {relative}")
        receipt = receipts.get(relative)
        if receipt and destination.exists() and destination.stat().st_size == receipt["bytes"]:
            if hashlib.sha256(destination.read_bytes()).hexdigest() == receipt["sha256"]:
                return receipt
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(6):
            try:
                if not getattr(local, "connection", None):
                    local.connection = http.client.HTTPSConnection("amazon-berkeley-objects.s3.us-east-1.amazonaws.com", timeout=60)
                connection = local.connection
                connection.request("GET", "/" + relative)
                response = connection.getresponse()
                data = response.read()
                if response.status != 200:
                    raise OSError(f"HTTP {response.status} for {relative}")
                if len(data) != int(response.getheader("Content-Length")):
                    raise OSError(f"Truncated image: {relative}")
                etag = response.getheader("ETag", "").strip('"')
                if len(etag) == 32 and hashlib.md5(data).hexdigest() != etag:
                    raise OSError(f"Source ETag mismatch: {relative}")
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(destination)
                return {"path": relative, "url": BASE_URL + relative, "bytes": len(data),
                        "etag": etag, "sha256": hashlib.sha256(data).hexdigest()}
            except (OSError, http.client.HTTPException):
                if getattr(local, "connection", None):
                    local.connection.close()
                local.connection = None
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("Image download failed")

    with ledger.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for count, receipt in enumerate(executor.map(fetch, sorted(set(relative_paths))), 1):
                output.write(json.dumps(receipt) + "\n")
                output.flush()
                if count % 500 == 0:
                    print(f"Verified/downloaded {count:,}/{len(set(relative_paths)):,} main images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/abo"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full-image-archive", action="store_true", help="Also download all catalog images as a 3 GB archive")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    for source in FILES:
        name = "SOURCE_README.md" if source == "README.md" else Path(source).name
        download(source, args.out / name, args.workers)
    if args.full_image_archive:
        source = "archives/abo-images-small.tar"
        download(source, args.out / Path(source).name, args.workers)


if __name__ == "__main__":
    main()
