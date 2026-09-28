"""Differentiable Owen's T function for PyTorch.

This module provides a PyTorch autograd-compatible implementation of Owen's
T function, which appears in the exact variance expressions for arbitrary-
level crossings (Theorems III.1 and III.2 of the paper).  The forward pass wraps
``scipy.special.owens_t``, and analytical gradients are provided for
backpropagation through the variance integrand.
"""

import math

import scipy.special
import torch
import torch.special


class OwensT(torch.autograd.Function):
    """PyTorch autograd Function for Owen's T function.

    Owen's T function T(h, a) is defined as:
        T(h, a) = (1/2π) ∫₀ᵃ exp(-h²(1+t²)/2) / (1+t²) dt

    This implementation wraps scipy.special.owens_t and provides
    analytical gradients for backpropagation.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, h: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        """Compute Owen's T function.

        Args:
            ctx: Context object for saving tensors for backward pass.
            h: First argument of Owen's T function.
            a: Second argument of Owen's T function.

        Returns:
            The value of Owen's T function T(h, a).
        """
        # Convert to NumPy arrays (detach to avoid tracking in autograd)
        h_np = h.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        # Call the SciPy implementation of Owen's T
        result = scipy.special.owens_t(h_np, a_np)
        # Save tensors for the backward pass
        ctx.save_for_backward(h, a)
        # Return as a torch tensor
        return torch.tensor(result, dtype=h.dtype, device=h.device)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients of Owen's T function.

        The analytical derivatives are:
            ∂T/∂h = -exp(-h²/2) * erf(a*h/√2) / (2√(2π))
            ∂T/∂a = exp(-h²(1+a²)/2) / (2π(1+a²))

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
            Tuple of gradients with respect to h and a.
        """
        h, a = ctx.saved_tensors

        # Analytical derivative with respect to h:
        # dT/dh = - exp(-h^2/2) * erf(a*h/sqrt(2)) / (2 * sqrt(2*pi))
        dT_dh = (
            -torch.exp(-(h**2) / 2)
            * torch.special.erf(a * h / math.sqrt(2))
            / (2 * math.sqrt(2 * math.pi))
        )

        # Analytical derivative with respect to a:
        # dT/da = exp(-h^2*(1+a^2)/2) / (2*pi*(1+a^2))
        dT_da = torch.exp(-(h**2) * (1 + a**2) / 2) / (2 * math.pi * (1 + a**2))

        # Chain rule: multiply with incoming gradient
        return (grad_output * dT_dh).sum_to_size(h.shape), (grad_output * dT_da).sum_to_size(
            a.shape
        )


def owensT(h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Compute Owen's T function with autograd support.

    Owen's T function T(h, a) is defined as:
        T(h, a) = (1/2π) ∫₀ᵃ exp(-h²(1+t²)/2) / (1+t²) dt

    This is a convenience function that applies the OwensT autograd Function.

    Args:
        h: First argument of Owen's T function.
        a: Second argument of Owen's T function.

    Returns:
        The value of Owen's T function T(h, a).

    Example:
        >>> h = torch.tensor(1.0, requires_grad=True)
        >>> a = torch.tensor(0.5, requires_grad=True)
        >>> result = owensT(h, a)
        >>> result.backward()
    """
    return OwensT.apply(h, a)
