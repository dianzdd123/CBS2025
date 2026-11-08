# Contributing to CBS2025

Start with the [reuse guide](docs/REUSE.md) and preserve the upstream publication credits and licence notices.

## Propose a reproducible change

Open an [issue](https://github.com/dianzdd123/CBS2025/issues) describing the expected and observed behaviour, commit SHA, operating system, Python/runtime versions and a minimal example. Include a small synthetic input where possible; do not attach credentials, private data or large model files.

For a pull request, keep the change focused. Explain what changed, why, the commands used to validate it and any effect on split membership, model inputs, output units or reported metrics. Update the README/reuse guide when the user workflow changes. Do not replace research results with numbers from an unvalidated environment.

## Checks before submission

```bash
python scripts/check_repository.py
```

This uses Python 3.10+ and does not run experiments. If `tests/` is present, run its focused suite as described below. GPU, simulator and hardware changes need separate validation on the matching platform; include that evidence or clearly state the limitation. Preserve vendor SDK source and historical data unless the change specifically requires modifying them.

### Focused tests

Install NumPy in a utility-test environment, then run `python -m unittest discover -s tests -v`. The tests load the actual experiment function definitions without running module-level data/training setup. Array-backed fixtures replace Torch I/O; see the reuse guide for the scope.
