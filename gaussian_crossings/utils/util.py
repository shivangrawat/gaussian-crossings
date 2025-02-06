import numpy as np
import torch


def count_upcrossings(x, threshold):
    """
    Count the number of upcrossings of a threshold in a 1D signal.

    Parameters
    ----------
    x : torch.Tensor
        1D signal.
    threshold : float
        Threshold value.

    Returns
    -------
    int
        Number of upcrossings.
    """
    upcrossings = torch.logical_and(x[:-1] < threshold, x[1:] >= threshold)
    return upcrossings.sum().item()