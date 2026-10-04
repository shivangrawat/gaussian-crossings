"""Freeze compatibility separately from independent mathematical checks."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gaussian_crossings import formula, process

CASES = json.loads((Path(__file__).parent / "fixtures/legacy_statistics.json").read_text())["cases"]


@pytest.mark.filterwarnings("ignore:Nonfinite small-lag samples:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:GaussianUpCrossingsDimless.*is deprecated:DeprecationWarning")
@pytest.mark.parametrize(
    "case", CASES, ids=lambda c: f"{c['class']}-{c['correlation']}-{c['parameters']}-u{c['u']}"
)
def test_pre_refactor_statistics(case):
    model = getattr(formula, case["class"])(
        getattr(process, case["correlation"]), u=case["u"], **case["parameters"]
    )
    lag = torch.tensor([0.1, 0.4, 1.0, 4.0], dtype=torch.float64)
    options = {**case["integration_options"], "method": "trapezoid"}
    actual = {
        "r0": model.r0,
        "q0": model.q0,
        "mean_rate": model.upcrossing_mean_rate(),
        "integrand": model.upcrossing_integrand(lag),
        "variance": model.upcrossing_variance(10.0, **options),
        "variance_rate": model.upcrossing_variance_CLT_per_unit_time(**options),
    }
    if "total_integrand" in case["expected"]:
        actual.update(
            total_integrand=model.crossing_integrand(lag),
            total_variance=model.crossing_variance(10.0, method="trapezoid"),
            total_variance_rate=model.crossing_variance_CLT_per_unit_time(method="trapezoid"),
            fano=model.upcrossing_fano_factor_CLT(method="trapezoid"),
        )
    for name, value in actual.items():
        np.testing.assert_allclose(
            value.detach().numpy(), case["expected"][name], rtol=2e-10, atol=2e-11, err_msg=name
        )
