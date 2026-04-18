# Contributing to slim-ml

Thanks for your interest. slim-ml is an adaptive LLM runtime focused on running
big models on constrained hardware. It's early-stage scaffold — contributions
that move one of the roadmap items from `NotImplementedError` to "works and
measured" are the most valuable.

## Roadmap priorities

See `README.md` for current status. In order:

1. **Routing hooks in `MLXBackend`** — unblocks `tools/routing_probe.py`
2. **Expert cache migration logic** — `src/slim_ml/technique.py::ExpertCache`
3. **`Tier.UNIFIED`** for Apple Silicon — replaces `Tier.VRAM=0` hack
4. **Speculative decoding, KV quantization, layer streaming** — interface stubs

If you want to work on something not on this list, open an issue first so we
can discuss fit.

## Development setup

```bash
git clone https://github.com/PewterZz/slim-ml.git
cd slim-ml
pip install -e ".[mlx,llama,dev]"   # or drop mlx/llama based on your platform
pytest
```

## Code style

- Keep modules small and interface-first. The `Technique` and `Backend` ABCs
  are load-bearing — extend them deliberately.
- Prefer explicit `NotImplementedError("reason")` over silent no-ops when a
  feature is planned but not built.
- No comments explaining what code does — names should do that. Comments are
  for *why* (hidden constraints, non-obvious invariants).
- Run `pytest` before pushing.

## Reporting bugs

Include:
- Hardware (VRAM, RAM, model/arch of your laptop or Mac)
- Model you were running
- Full traceback
- Output of `slim-ml probe`

## Pull requests

- One logical change per PR.
- Tests for new behavior; a failing test first if fixing a bug.
- Update `README.md` if you change the "what works" / "what's stub" status.
- PRs that add measurement (benchmarks, telemetry, probes) are especially
  welcome — this project optimizes for "can't improve what you can't measure."

## License

By contributing, you agree your contributions are licensed under the MIT
License (see `LICENSE`).
