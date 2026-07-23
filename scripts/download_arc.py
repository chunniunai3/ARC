"""Download ARC-AGI datasets."""

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def clone_repo(url: str, dest: Path, name: str) -> None:
    if dest.exists():
        print(f"✓ {name} already exists at {dest}")
        return
    print(f"Cloning {name} from {url} ...")
    result = subprocess.run(
        ["git", "clone", url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"✓ {name} downloaded to {dest}")
    else:
        print(f"✗ Failed to clone {name}: {result.stderr}")


if __name__ == "__main__":
    repos = {
        "ARC-AGI-1": "https://github.com/fchollet/ARC-AGI.git",
        "ARC-AGI-2": "https://github.com/fchollet/ARC-AGI-2.git",
        "NSA": "https://github.com/Batorskq/NSA.git",
    }
    for name, url in repos.items():
        clone_repo(url, DATA_DIR / name, name)

    print("\nDone. Run the following to verify:")
    print(f"  {sys.executable} -c \"import arc; print('arc-py OK')\"")
