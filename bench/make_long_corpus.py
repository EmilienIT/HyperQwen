"""Build ~/bench/labd_corpus_long.txt, the document the long-context LABD runs use.

The frozen corpus (~/bench/labd_corpus.txt, built by labd_bench.py on first run) is this
repo's own docs repeated until it reaches 200k characters -- about 84k tokens. Two problems
past ~65k of context: labd_bench.py slices `corpus[:ctx * 3.6]` and would silently measure a
shorter prompt than asked for, and the repetition hands the suffix lookup free matches from
an earlier copy of the same text, which flatters exactly the number the lookup is judged on.

This keeps the frozen corpus as the head -- byte-identical, so every number taken at
--ctx 20000 stays comparable -- and appends vLLM's own source. That is varied (no repeats)
and it is the realistic shape for this mode anyway: a coding assistant with a large real
codebase in context.

  venv/bin/python bench/make_long_corpus.py
  venv/bin/python bench/labd_bench.py <tag> --ctx 100000 --corpus ~/bench/labd_corpus_long.txt

Note the corpus runs ~2.9 characters per token, not the 3.6 labd_bench.py assumes, so
--ctx N produces roughly 1.24 N tokens. --ctx 100000 gave 112,655-token prompts.
"""
import glob
import importlib.util
import os
import sys

BASE = os.path.expanduser("~/bench/labd_corpus.txt")
OUT = os.path.expanduser("~/bench/labd_corpus_long.txt")
TARGET = 900_000  # characters of filler, on top of the frozen head

if not os.path.exists(BASE):
    sys.exit(f"{BASE} not found -- run bench/labd_bench.py once to build the frozen corpus")

with open(BASE, encoding="utf-8", newline="") as f:
    base = f.read()
# Find the active install without importing vLLM (which may initialize CUDA).
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    sys.exit("vLLM is not installed in this Python environment; run with the serving venv's Python")
src = os.path.join(next(iter(spec.submodule_search_locations)), "v1")
files = sorted(glob.glob(os.path.join(src, "**", "*.py"), recursive=True))
if not files:
    sys.exit(f"no vLLM source under {src}")

parts, seen = [], 0
for f in files:
    try:
        text = open(f, encoding="utf-8", errors="ignore").read()
    except OSError:
        continue
    if len(text) < 2000:
        continue
    parts.append("\n\n### " + os.path.basename(f) + "\n\n" + text)
    seen += len(text)
    if seen > TARGET:
        break

long = base + "".join(parts)
assert long[: len(base)] == base, "the frozen head must stay byte-identical"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="") as f:
    f.write(long)
print(f"{OUT}: {len(long)} chars ({len(base)} frozen head + {len(parts)} source files)")
