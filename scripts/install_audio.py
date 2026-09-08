"""Fetch a pinned official OpenAL Soft binary into this checkout, not Windows.

SHA256 is the digest observed for the upstream 1.25.2 ZIP on 2026-09-08;
it provides reproducibility, not an independently signed publisher attestation.
"""
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import zipfile

VERSION = "1.25.2"
URL = f"https://openal-soft.org/openal-binaries/openal-soft-{VERSION}-bin.zip"
SHA256 = "67a0c4b800bd860c93c04f38caf8cbe4875f9c84700ac430efc451f70e265434"
ROOT = Path(__file__).resolve().parents[1]


def main():
    if struct.calcsize("P") != 8:
        raise SystemExit("This prototype installer requires a 64-bit Python interpreter.")
    local = ROOT / "local"
    local.mkdir(exist_ok=True)
    archive = local / f"openal-soft-{VERSION}-bin.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise SystemExit(f"Checksum mismatch; not installing. Inspect {temporary}.")
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise SystemExit(f"Checksum mismatch; not installing. Inspect {archive}.")
    target = local / "openal-soft"
    # Explicit members only: no executable installer, arbitrary paths or system writes.
    members = ["bin/Win64/soft_oal.dll", "COPYING"]
    with zipfile.ZipFile(archive) as bundle:
        prefix = f"openal-soft-{VERSION}-bin/"
        for name in members:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bundle.read(prefix + name))
    manifest = {"version": VERSION, "url": URL, "archive_sha256": SHA256,
                "members": {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in members}}
    (target / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Installed {target / members[0]}")


if __name__ == "__main__":
    main()
