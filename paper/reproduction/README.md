# PRE revision archive

This directory (and the tag `pre-revision-2026-09-27`) holds the saved numerical
inputs for Figures 2–6 of the revised manuscript. `pre_revision_data.zip` contains the
analytical arrays, all 50,000 trajectories' crossing counts and covariance
probes, the Figure 3 bootstrap summary, and the original source snapshots.
It does not store full sampled trajectories. The original numerical
metadata are preserved. `manifest.json` records SHA-256 checksums and the
comparison of numerical function definitions with the maintained scripts.

From the repository root:

```sh
uv sync --all-extras --locked
uv run python paper/restore_pre_archive.py
uv run python paper/pre_figures.py plot \
  --output data/pre_figures_20260927 --figures figures/publication
```

Restoration checks every file before writing and refuses to overwrite
different local data. Use `--destination /path/to/empty/data` for an isolated
copy, or `--verify-only` to check without writing. The renderer uses the
default Figure 3 archive under `data/pre_figure3_10000`.

The maintained scripts differ from the archived snapshots in plotting and
command handling; their numerical functions are identical. The bundle
preserves the original snapshots because Figure 3 reanalysis checks the
complete source hash:

```sh
uv run python data/pre_figure3_10000/run_source.py analyze \
  --output data/pre_figure3_10000 --bootstrap 10000
```

The current renderer also exports alternative illustrations for Figure 1
and Supplemental Figure S1. The submitted revision retains the original
illustrations; only Figures 2–6 are covered by this numerical archive.
Recalculation and new simulations are described in
[the reproduction guide](../../docs/reproducibility.md).
