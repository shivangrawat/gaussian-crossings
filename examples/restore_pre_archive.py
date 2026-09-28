"""Restore the published PRE numerical archives without overwriting local work."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def restore(destination, verify_only=False):
    manifest = json.loads((ROOT / "reproduction/manifest.json").read_text())
    archive = ROOT / "reproduction" / manifest["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Published archive checksum mismatch")
    pending = []
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != set(manifest["files"]):
            raise ValueError("Archive file list differs from the manifest")
        for name, record in manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid archive path: {name}")
            content = bundle.read(name)
            if (len(content) != record["bytes"]
                    or hashlib.sha256(content).hexdigest() != record["sha256"]):
                raise ValueError(f"File checksum mismatch: {name}")
            target = destination / relative
            if target.exists():
                if not target.is_file() or target.read_bytes() != content:
                    raise FileExistsError(f"Refusing to replace different local data: {target}")
            else:
                pending.append((target, content))
    if not verify_only:
        for target, content in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
    print(f"Verified {len(manifest['files'])} files; "
          f"{'would restore' if verify_only else 'restored'} {len(pending)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    restore(args.destination, args.verify_only)


if __name__ == "__main__":
    main()
