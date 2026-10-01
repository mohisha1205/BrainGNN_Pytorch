"""
Patch the installed `deepdish` package so it imports under NumPy 2.x.

deepdish 0.3.7 still references aliases that NumPy 2 removed
(np.ComplexWarning, np.unicode_, np.string_, np.object). This script
rewrites them in place inside site-packages. Run it once after
`pip install -r requirements.txt`:

    python scripts/patch_deepdish.py

It is the portable equivalent of the `sed` commands in the Colab notebook.
"""
import importlib.util
import os
import re
import sys

spec = importlib.util.find_spec("deepdish")
if spec is None or not spec.submodule_search_locations:
    sys.exit("deepdish is not installed - run `pip install deepdish` first.")

pkg_dir = list(spec.submodule_search_locations)[0]

REPLACEMENTS = [
    (re.compile(r"np\.unicode_\b"), "np.str_"),
    (re.compile(r"np\.string_\b"), "np.bytes_"),
    (re.compile(r"np\.object\b(?!_)"), "object"),
]

patched = 0
for root, _, files in os.walk(pkg_dir):
    for name in files:
        if not name.endswith(".py"):
            continue
        path = os.path.join(root, name)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new = text
        # Drop lines that reference the removed np.ComplexWarning
        if name == "core.py":
            new = "\n".join(l for l in new.split("\n") if "np.ComplexWarning" not in l)
        for pattern, repl in REPLACEMENTS:
            new = pattern.sub(repl, new)
        if new != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            patched += 1
            print(f"patched {path}")

print(f"Done - {patched} file(s) patched in {pkg_dir}")
