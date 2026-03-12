# Contributing

Thanks for contributing to `llm-expose`.

## Development Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install project and development dependencies:
   - `pip install -e .[dev]`
3. Run tests:
   - `pytest`

## Code Quality

Run these checks before opening a PR:

- `ruff check .`
- `black --check .`
- `mypy llm_expose`
- `pytest`

## Pull Request Guidelines

- Keep PRs focused and small when possible.
- Add or update tests for behavior changes.
- Update docs for user-facing changes.
- Add a changelog entry under `Unreleased` (or update release notes flow when introduced).

## Versioning

This project follows Semantic Versioning (`MAJOR.MINOR.PATCH`).

- `PATCH`: bug fixes and non-breaking improvements
- `MINOR`: new backward-compatible features
- `MAJOR`: backward-incompatible changes

## Reporting Issues

Please include:

- Reproduction steps
- Expected behavior
- Actual behavior
- Logs or stack traces when available
- Environment details (OS, Python version, provider/channel config)
