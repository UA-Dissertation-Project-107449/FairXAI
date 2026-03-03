# FairXAI Documentation & Structure Style Guide

This guide defines the minimum documentation and structure expectations for
modules under `src/fairxai`.

## 1) Module-Level Requirements

Each module folder (e.g., `fairness/`, `models/`) must include:

- `README.md` with:
  - Purpose (what this module does)
  - File overview table
  - Key classes/functions
  - Configuration dependencies
  - Short usage example
  - External dependencies (when relevant)

## 2) Python File-Level Requirements

Every Python file should have a top module docstring that includes:

- What the file is responsible for
- Main public symbols (classes/functions)
- Any important runtime assumptions

## 3) Class & Function Docstrings

### Classes

Class docstrings should cover:

- Purpose
- Important constructor arguments
- Behavioral notes when needed

### Functions

Public functions should document:

- Purpose
- Args
- Returns
- Raises (if applicable)
- Notes/examples for non-trivial behavior

## 4) API Surface & Exports

- `__init__.py` should expose the intended public API explicitly via `__all__`.
- Keep package docstrings concise and module-oriented.

## 5) Configuration Principles

- Prefer YAML config over hardcoded tunables or ad-hoc environment variables.
- Document config paths in module README files.
- Keep behavior defaults explicit and visible in code.

## 6) Logging Principles

- Use `logging.getLogger(__name__)` in modules.
- Avoid `print()` in package code.
- Keep log messages concise and actionable.

## 7) Readability & Consistency

- Keep naming consistent across modules (e.g., `*_summary`, `*_metrics`).
- Prefer type hints for public functions.
- Favor small, composable helpers over deeply nested monoliths.

## 8) Stubs and Planned Features

If functionality is intentionally scaffolded:

- Raise `NotImplementedError` with clear intent.
- Include roadmap note with target window in module docstring and README.

## 9) Change Safety Checklist

For docs-only module updates:

1. Add/update README
2. Update package/file docstrings if needed
3. Run syntax check on touched Python files
4. Keep runtime behavior unchanged unless explicitly requested

## 10) Review Checklist

Before marking module docs complete:

- [ ] Module has README
- [ ] `__init__.py` docstring reflects public API role
- [ ] Public APIs are explained at least once (README or docstrings)
- [ ] Config dependencies are documented
- [ ] Any stubs are explicitly marked with roadmap notes
