# Contributing

Thanks for contributing to `solvr`.

## Prerequisites

- Rust toolchain compatible with MSRV (`1.85`) or newer.
- A clean working tree before opening a pull request.

## Setup

```bash
cargo check
cargo test --lib
```

## Local Quality Checks

Run these before submitting:

```bash
cargo fmt --all -- --check
cargo check --all-features
cargo test --lib
```

If you are working on lint cleanup, also run:

```bash
cargo clippy --all-targets
```

## Pull Request Guidelines

- Keep PRs focused and scoped.
- Include tests for behavioral changes.
- Update docs when public APIs or features change.
- Add a short summary of what changed and why.

## Commit Messages

Use clear, imperative messages (for example: `add sparse graph validation for bellman-ford`).
