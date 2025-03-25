import torch
import torch.special
import scipy.special
import math


class OwensT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, a):
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
    def backward(ctx, grad_output):
        h, a = ctx.saved_tensors

        # Analytical derivative with respect to h:
        # dT/dh = - exp(-h^2/2) * erf(a*h/sqrt(2)) / (2 * sqrt(2*pi))
        dT_dh = - torch.exp(-h**2 / 2) * torch.special.erf(a * h / math.sqrt(2)) / (2 * math.sqrt(2 * math.pi))

        # Analytical derivative with respect to a:
        # dT/da = exp(-h^2*(1+a^2)/2) / (2*pi*(1+a^2))
        dT_da = torch.exp(-h**2 * (1 + a**2) / 2) / (2 * math.pi * (1 + a**2))

        # Chain rule: multiply with incoming gradient
        return grad_output * dT_dh, grad_output * dT_da

# Convenience function to call our custom OwenT
def owensT(h, a):
    return OwensT.apply(h, a)
