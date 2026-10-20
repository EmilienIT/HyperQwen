"""The frozen document shared by the LABD benchmarks. Stdlib only."""
from pathlib import Path

import harness


def load_corpus(path):
    """Reuse an existing corpus, or freeze this checkout's documentation on first run."""
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8")

    repo = Path(harness.REPO)
    files = sorted([*repo.glob("*.md"), *repo.glob("*/README.md")])
    text = "\n\n".join(p.read_text(encoding="utf-8") for p in files)
    if not text.strip():
        raise ValueError(f"no documentation found in {repo}; cannot build LABD corpus {path}")
    document = text
    while len(text) < 200000:
        text += "\n\n" + document
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
