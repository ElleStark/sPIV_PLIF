# Contributing

Thanks for your interest in contributing to `sPIV_PLIF`! This document gives a quick, practical guide for setting up a development environment, running tests, building docs, and submitting changes.

1. Quick start
- Clone the repository:
  ```powershell
  git clone https://github.com/ElleStark/sPIV_PLIF.git
  cd sPIV_PLIF
  ```
- Create and activate a virtual environment (Windows PowerShell):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- Install editable package plus development extras:
  ```powershell
  python -m pip install --upgrade pip
  python -m pip install -e .[dev,docs]
  ```

2. Test and lint
- Run unit tests with pytest:
  ```powershell
  pytest
  ```
- Run formatter and linters (Black and Ruff):
  ```powershell
  black src tests
  ruff check src tests
  ```
- We recommend enabling the provided `pre-commit` hooks:
  ```powershell
  pre-commit install
  pre-commit run --all-files
  ```

3. Building docs locally
- Install the `docs` extras (done above) then build with Sphinx:
  ```powershell
  python -m sphinx -b html docs/source docs/_build/html
  ```

4. Package layout & adding new modules
- Python package sources live under `src/` and the main package is `sPIV_PLIF_postprocessing`.
- Create new modules inside the appropriate subpackage (`analysis`, `io`, `utils`, `visualization`). Add tests under `tests/` mirroring the package path.

5. Commit and PR workflow
- Keep changes small and focused; one logical change per branch/PR.
- Use descriptive commit messages. Example:
  ```text
  feat(io): add TIFF reader with tests
  ```
- Open a pull request against `main`. Link any relevant issue and describe the change, testing steps, and any backwards-incompatible behavior.

6. Style and guidelines
- Follow standard Python idioms and PEP 8. Use `black` for formatting.
- Avoid heavy side-effects in package `__init__.py` files; prefer lazy-loading for optional heavy modules.

7. CI and docs hosting
- CI (GitHub Actions) runs tests, linting, and docs builds. If you modify build steps, update the workflow files under `.github/workflows`.

8. Questions
- If you're unsure where to put code or need to discuss API design, open an issue or a draft PR and ask for feedback.

Thanks — we appreciate your contributions!
