"""SDE-based dynamical system models for stochastic processes.

This module provides classes representing stochastic dynamical systems
whose sample paths can be generated via Euler-Maruyama integration.  These
models are used for numerical validation of the exact variance and Fano
factor formulae by comparing simulation-based crossing counts against the
analytical predictions.
"""

import math
from typing import Optional

import torch

from gaussian_crossings.utils.utils import dynm_fun


class filtered_OU:
    """Filtered Ornstein-Uhlenbeck process model.

    This class represents an OU process x(t) that has been passed through a
    first-order low-pass filter to produce y(t).  The resulting two-dimensional
    system has a bi-exponential correlation function for y(t).

    Attributes:
        tau_e: Time constant of the OU process x(t).
        tau_f: Time constant of the low-pass filter.
        sigma: Noise strength (standard deviation of x(t)).
        dim: Dimension of the state space (always 2: x and y).
        J: Jacobian matrix of the linearized system.
    """

    def __init__(self, tau_e: float = 1.0, tau_f: float = 1.0, sigma: float = 1.0) -> None:
        """Initialize the filtered OU process.

        Args:
            tau_e: Effective time constant of the OU process.
            tau_f: Filter time constant.
            sigma: Noise strength (standard deviation).
        """
        self.tau_e = tau_e
        self.tau_f = tau_f
        self.sigma = sigma
        self.dim = 2

        # Define the Jacobian matrix
        self.J = torch.tensor(
            [[-1 / self.tau_e, 0], [1 / self.tau_f, -1 / self.tau_f]], dtype=torch.float64
        )

    def noise_vector(self) -> torch.Tensor:
        """Return the noise diffusion vector.

        The noise acts only on the first component (x).

        Returns:
            Tensor of shape (2,) containing the noise coefficients.
        """
        return torch.tensor([self.sigma * math.sqrt(2 / self.tau_e), 0.0], dtype=torch.float64)

    @dynm_fun
    def _dynamical_fun(self, t: Optional[float], vars: torch.Tensor) -> torch.Tensor:
        """Compute the drift term of the SDE.

        The dynamics are:
            dx/dt = -x / tau_e
            dy/dt = (x - y) / tau_f

        Args:
            t: Time (unused, included for API compatibility).
            vars: State vector of shape (2,) containing [x, y].

        Returns:
            Time derivative [dx/dt, dy/dt] as a tensor of shape (2,).
        """
        vars = vars.squeeze(0)
        x = vars[0:1]
        y = vars[1:]
        dxdt = (1 / self.tau_e) * (-x)
        dydt = (1 / self.tau_f) * (-y + x)
        return torch.cat((dxdt, dydt))

    def steady_state(self) -> torch.Tensor:
        """Return the steady-state (equilibrium) of the system.

        Returns:
            Tensor of shape (2,) containing the equilibrium state [0, 0].
        """
        return torch.tensor([0.0, 0.0], dtype=torch.float64)


class OU_noise:
    """Mean-reverting process driven by Ornstein-Uhlenbeck noise (Section IV.B).

    This class represents the two-dimensional system from Section IV.B of
    the paper, where a mean-reverting process y(t) is driven by an OU
    process x(t).  The ratio kappa = tau_f / tau_e controls how filtered
    the noise appears to the system and determines the crossing statistics.

    Attributes:
        tau_e: Relaxation time constant of the mean-reverting process y(t).
        tau_f: Correlation time of the driving OU noise x(t).
        sigma: Noise strength (standard deviation of x(t)).
        dim: Dimension of the state space (always 2: x and y).
        J: Jacobian matrix of the linearized system.
    """

    def __init__(self, tau_e: float = 1.0, tau_f: float = 1.0, sigma: float = 1.0) -> None:
        """Initialize the OU noise process.

        Args:
            tau_e: Effective time constant of the main process.
            tau_f: Time constant of the OU noise.
            sigma: Noise strength (standard deviation).
        """
        self.tau_e = tau_e
        self.tau_f = tau_f
        self.sigma = sigma
        self.dim = 2

        # Define the Jacobian matrix
        self.J = torch.tensor(
            [[-1 / self.tau_f, 0], [1 / self.tau_e, -1 / self.tau_e]], dtype=torch.float64
        )

    def noise_vector(self) -> torch.Tensor:
        """Return the noise diffusion vector.

        The noise acts only on the first component (x).

        Returns:
            Tensor of shape (2,) containing the noise coefficients.
        """
        return torch.tensor([self.sigma * math.sqrt(2 / self.tau_f), 0.0], dtype=torch.float64)

    @dynm_fun
    def _dynamical_fun(self, t: Optional[float], vars: torch.Tensor) -> torch.Tensor:
        """Compute the drift term of the SDE.

        The dynamics are:
            dx/dt = -x / tau_f
            dy/dt = (x - y) / tau_e

        Args:
            t: Time (unused, included for API compatibility).
            vars: State vector of shape (2,) containing [x, y].

        Returns:
            Time derivative [dx/dt, dy/dt] as a tensor of shape (2,).
        """
        vars = vars.squeeze(0)
        x = vars[0:1]
        y = vars[1:]
        dxdt = (1 / self.tau_f) * (-x)
        dydt = (1 / self.tau_e) * (-y + x)
        return torch.cat((dxdt, dydt))

    def steady_state(self) -> torch.Tensor:
        """Return the steady-state (equilibrium) of the system.

        Returns:
            Tensor of shape (2,) containing the equilibrium state [0, 0].
        """
        return torch.tensor([0.0, 0.0], dtype=torch.float64)


class damped_harmonic_oscillator_noise:
    """Stochastic damped harmonic oscillator model (Section IV.A of the paper).

    This class represents a damped harmonic oscillator driven by thermal
    white noise, following the Langevin equation:
        d²x/dt² + 2*zeta*omega0*dx/dt + omega0²*x = sqrt(4*zeta*omega0*T)*eta(t)

    where eta(t) is Gaussian white noise.  The noise amplitude satisfies the
    fluctuation-dissipation theorem, ensuring a stationary variance of
    r(0) = temp / omega0^2.  The damping ratio zeta controls whether the
    process exhibits oscillatory (underdamped) or monotonic (overdamped)
    correlations. Crossing-count dispersion depends jointly on this
    damping and on the chosen threshold.

    Attributes:
        zeta: Damping ratio (zeta < 1: underdamped, zeta = 1: critical,
            zeta > 1: overdamped).
        omega0: Natural angular frequency.
        temp: Temperature (noise strength, with k_B=1).
        dim: Dimension of the state space (always 2: position and velocity).
        J: Jacobian matrix of the linearized system.
    """

    def __init__(self, zeta: float = 0.5, omega0: float = 1.0, temp: float = 1.0) -> None:
        """Initialize the damped harmonic oscillator.

        Args:
            zeta: Damping ratio.
            omega0: Natural angular frequency.
            temp: Temperature (noise strength).
        """
        self.zeta = zeta
        self.omega0 = omega0
        self.temp = temp
        self.dim = 2

        # Jacobian of the linearized system
        self.J = torch.tensor(
            [[0, 1], [-(self.omega0**2), -2 * self.zeta * self.omega0]], dtype=torch.float64
        )

    def noise_vector(self) -> torch.Tensor:
        """Return the noise diffusion vector.

        The noise acts only on the velocity component (second element).

        Returns:
            Tensor of shape (2,) containing the noise coefficients.
        """
        return torch.tensor(
            [0.0, math.sqrt(4 * self.zeta * self.omega0 * self.temp)], dtype=torch.float64
        )

    @dynm_fun
    def _dynamical_fun(self, t: Optional[float], vars: torch.Tensor) -> torch.Tensor:
        """Compute the drift term of the SDE.

        The dynamics are:
            dx/dt = v
            dv/dt = -2*zeta*omega0*v - omega0²*x

        Args:
            t: Time (unused, included for API compatibility).
            vars: State vector of shape (2,) containing [x, v].

        Returns:
            Time derivative [dx/dt, dv/dt] as a tensor of shape (2,).
        """
        vars = vars.squeeze(0)
        x = vars[0:1]
        y = vars[1:]
        dxdt = y
        dydt = -2 * self.zeta * self.omega0 * y - (self.omega0**2) * x
        return torch.cat((dxdt, dydt))

    def steady_state(self) -> torch.Tensor:
        """Return the steady-state (equilibrium) of the system.

        The equilibrium is at rest with zero displacement and velocity.

        Returns:
            Tensor of shape (2,) containing the equilibrium state [0, 0].
        """
        return torch.tensor([0.0, 0.0], dtype=torch.float64)
