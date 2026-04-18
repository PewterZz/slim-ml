# Security Policy

## Supported Versions

slim-ml is in early development. Only the latest `main` branch receives security fixes.

## Reporting a Vulnerability

If you discover a security issue, please **do not open a public GitHub issue**.

Instead, report privately via GitHub's [Security Advisories](https://github.com/PewterZz/slim-ml/security/advisories/new) form.

Please include:
- A description of the issue and its impact
- Steps to reproduce (minimal repro preferred)
- Affected versions / commit SHAs
- Any suggested mitigations

You can expect an initial response within 7 days. Fixes for confirmed issues will be coordinated with you before public disclosure.

## Scope

slim-ml loads and runs third-party model weights. Issues in upstream model code (`mlx-lm`, `llama-cpp-python`, tokenizers) should be reported to their respective maintainers. Issues in slim-ml's own code — the runtime, backend abstractions, telemetry, or CLI — are in scope.
