"""Hand-counted boundary cases and independently calculated special functions."""

import numpy as np
import pytest
import torch
from scipy.special import owens_t

from gaussian_crossings.utils import (
    autocorrelation,
    count_crossings,
    count_downcrossings,
    count_upcrossings,
    crossing_times,
    downcrossing_times,
    owensT,
    upcrossing_times,
)


@pytest.mark.parametrize(
    "values,ups,downs",
    [([-1, 0, 0, 1, 0, -1], 1, 1), ([-1, 1, -1, 1], 2, 1), ([], 0, 0), ([0], 0, 0)],
)
def test_counts_and_times_use_the_same_threshold_convention(values, ups, downs):
    x = torch.tensor(values, dtype=torch.float64)
    t = torch.arange(len(values), dtype=torch.float64)
    assert count_upcrossings(x, 0) == ups
    assert count_downcrossings(x, 0) == downs
    assert count_crossings(x, 0) == ups + downs
    assert upcrossing_times(x, t, 0).numel() == ups
    assert downcrossing_times(x, t, 0).numel() == downs
    times, directions = crossing_times(x, t, 0)
    assert len(times) == ups + downs
    assert torch.all(times[1:] >= times[:-1])
    assert int((directions == 1).sum()) == ups


def test_linear_interpolation_and_multiple_levels():
    x = torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64)
    t = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    times, directions = crossing_times(x, t, 0.0)
    torch.testing.assert_close(times, torch.tensor([1.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(directions, torch.tensor([1.0, -1.0], dtype=torch.float64))
    assert count_crossings(x, [0.0, 2.0]) == [2, 0]


def test_covariance_normalization():
    x = np.array([1.0, 2.0, 4.0, 2.0])
    reference = np.array(
        [
            sum((x[i] - x.mean()) * (x[i + k] - x.mean()) for i in range(len(x) - k)) / (len(x) - 1)
            for k in range(len(x))
        ]
    )
    np.testing.assert_allclose(autocorrelation(x), reference)
    assert autocorrelation(x)[0] == pytest.approx(np.var(x, ddof=1))
    with pytest.raises(ValueError):
        autocorrelation([1.0])


def test_owens_t_values_and_signed_arguments():
    h = torch.tensor([-3.0, -0.1, 0.0, 0.1, 3.0], dtype=torch.float64)
    a = torch.tensor([0.2, -0.5, 1.0, 0.5, -0.2], dtype=torch.float64)
    np.testing.assert_allclose(owensT(h, a), owens_t(h.numpy(), a.numpy()), rtol=1e-14)
    torch.testing.assert_close(owensT(-h, a), owensT(h, a))
    torch.testing.assert_close(owensT(h, -a), -owensT(h, a))
