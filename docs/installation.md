# Installation

Python 3.10 or newer is required.

```bash
pip install gaussian-crossings
```

!!! note "Before the first PyPI release"
    If `pip` cannot find the package yet, install the latest version from GitHub:

    ```bash
    pip install "git+https://github.com/shivangrawat/gaussian-crossings"
    ```

The core dependencies are NumPy, SciPy, and PyTorch. PyTorch differentiates covariance
functions to obtain $r'(t)$ and $r''(t)$.

## A smaller install on Linux

The default PyTorch wheels on Linux include GPU libraries the package does not use. Installing
the CPU-only build first keeps the environment much smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gaussian-crossings
```

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `plot` | matplotlib | `gaussian_crossings.plotting` |
| `notebooks` | JupyterLab, matplotlib, and notebook tools | running the [tutorials](tutorials/README.md) |
| `docs` | MkDocs and its plugins | building this documentation |
| `dev` | pytest, ruff, matplotlib | running the tests |
| `reproduce` | mpmath, matplotlib | [reproducing the paper figures](reproducibility.md) |

For example, `pip install "gaussian-crossings[notebooks]"`.

## From a clone

```bash
git clone https://github.com/shivangrawat/gaussian-crossings.git
cd gaussian-crossings
uv sync --all-extras          # or: pip install -e ".[dev,notebooks,docs,reproduce]"
uv run pytest                 # or: python -m pytest
```

Preview the documentation locally with `uv run mkdocs serve`, or
`python -m mkdocs serve` after the pip installation above.
