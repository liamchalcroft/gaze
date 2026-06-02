# Contributing

## Dev quickstart

```bash
git clone https://github.com/liamchalcroft/gaze.git
cd gaze
uv sync              # install dependencies (includes the dev group)
make check           # lint, format, typecheck, lockfile, core tests
make check-nova      # torch-gated and example tests (installs the nova extra)
```

`make check` matches CI. Run `make check-nova` before touching anything that
the NOVA example or torch-dependent code paths exercise.

## House rules

- **Runtime validation**: put `@beartype` on public functions and class methods.
- **Exceptions**: raise specific types from `gaze.exceptions`. Never bare
  `except:` or broad `except Exception:`.
- **Commits**: conventional format -- `feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `perf:`.
- **Tests**: pytest. Mock external API calls; unit tests must not hit the
  network. Keep changed lines covered.

## Full guide

See [CONTRIBUTING.md](https://github.com/liamchalcroft/gaze/blob/main/CONTRIBUTING.md)
for the complete workflow, including new model adapters, new tools, evaluation
metrics, and pull-request expectations.
