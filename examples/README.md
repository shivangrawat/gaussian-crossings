# Tutorials

Short notebooks that run in seconds on a laptop. Every dataset is artificial and generated
inside the notebook with a fixed random seed, so results are reproducible and can be checked
against the known model.

| Notebook | What it shows |
|---|---|
| [01_quickstart](01_quickstart.ipynb) | Mean, variance, and Fano factor at one threshold and across thresholds; up-, down-, and total crossings; finite windows versus the long-time limit |
| [02_custom_covariance](02_custom_covariance.ipynb) | Writing your own covariance function, its requirements, and the errors raised by common mistakes |
| [03_fano_from_data](03_fano_from_data.ipynb) | Estimating the Fano factor from a recording with confidence intervals, comparing it with the finite-window prediction, and checking for signs of coarse sampling |
| [04_model_discrimination](04_model_discrimination.ipynb) | Recovering a hidden damping ratio from crossing counts when every candidate has the same mean crossing rate |

Install the notebook dependencies with `pip install "gaussian-crossings[notebooks]"`, or
`uv sync --extra notebooks` from a clone. The notebooks that reproduce the figures of the paper
are in [`paper/`](https://github.com/shivangrawat/gaussian-crossings/tree/main/paper).
