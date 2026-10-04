# PRE revision archive

This directory (and the tag `pre-revision-2026-09-27`) holds the saved numerical
inputs for Figures 2–6 of the revised manuscript. `pre_revision_data.zip` contains the
analytical arrays, all 50,000 trajectories' crossing counts and covariance
probes, the Figure 3 bootstrap summary, and the original source snapshots.
It does not store full sampled trajectories. The original numerical
metadata are preserved. `manifest.json` records SHA-256 checksums and the
source comparison made when the archive was prepared; its `current` paths
and hashes describe that historical snapshot.

From the repository root:

```sh
uv sync --all-extras --locked
uv run python paper/restore_pre_archive.py
uv run python paper/pre_figures.py plot \
  --output data/pre_figures_20260927 --figures figures/publication --historical
```

Restoration checks every file before writing and refuses to overwrite
different local data. Use `--destination /path/to/empty/data` for an isolated
copy, or `--verify-only` to check without writing. The renderer uses the
default Figure 3 archive under `data/pre_figure3_10000`.

The maintained runner now computes numerical results through the public
package, so its numerical functions differ from the archived snapshot.
`--historical` verifies the preserved source and array checksums and exports
the original arrays without recalculating them. Fresh calculations can be
compared numerically with these arrays using the reproduction guide below.
The bundle preserves the original Figure 3 source because reanalysis checks
its complete source hash:

```sh
uv run python data/pre_figure3_10000/run_source.py analyze \
  --output data/pre_figure3_10000 --bootstrap 10000
```

The current renderer also exports alternative illustrations for Figure 1
and Supplemental Figure S1. The submitted revision retains the original
illustrations; only Figures 2–6 are covered by this numerical archive.
Recalculation and new simulations are described in
[the reproduction guide](../../docs/reproducibility.md).
