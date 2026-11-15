# Contributing

Use Python 3.10 or newer; CI runs the tests on Python 3.10–3.13. Keep pull
requests small: add a regression test when applicable, make the focused change,
then update examples or documentation affected by it. Automated tests and default
offline benchmarks must use local fake models without API keys or model requests.
Bounded real benchmark commands are documented in [benchmarks/README.md](benchmarks/README.md).

From the repository root, create a development environment and run checks in
this order:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m build
python -m twine check dist/*
python scripts/smoke_wheel.py --check-metadata dist/*.whl
```

Use Python 3.13 for the packaging smoke, matching CI. Install the built wheel
without extras in a fresh environment, then run from outside the checkout with
isolated imports:

```bash
repo="$PWD"
smoke_dir="$(mktemp -d)"
python3.13 -m venv "$smoke_dir/venv"
"$smoke_dir/venv/bin/python" -m pip install --no-index "$repo"/dist/*.whl
(cd "$smoke_dir" && "$smoke_dir/venv/bin/python" -I "$repo/scripts/smoke_wheel.py")
```

The smoke checks public imports, local hygiene and compaction, version metadata,
`py.typed`, and that only the package and pip are installed. CI checks and builds
distributions; it does not publish them. Include the commands and results you
ran in your pull request, and explain any checks you could not run.
