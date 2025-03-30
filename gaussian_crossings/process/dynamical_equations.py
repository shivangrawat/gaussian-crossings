import torch 
import math
import numpy as np
from gaussian_crossings.utils.utils import dynm_fun


class filtered_OU:
    def __init__(self, tau_e=1.0, tau_f=1.0, sigma=1.0):
        # Define the model parameters (effective time constant, filtration time constant, and noise strength)
        self.tau_e = tau_e
        self.tau_f = tau_f
        self.sigma = sigma
        self.dim = 2

        # Define the jacobian
        self.J = torch.tensor([[-1 / self.tau_e, 0], [1 / self.tau_f, -1 / self.tau_f]])
    
    def noise_vector(self):
        # This function defines the noise vector for the filtered OU process.
        return torch.tensor([self.sigma * math.sqrt(2 / self.tau_e), 0.])

    @dynm_fun
    def _dynamical_fun(self, t, vars):
        # This function defines the dynamics of the filtered OU process.
        vars = vars.squeeze(0)  # Remove the extra dimension
        x = vars[0:1]
        y = vars[1:]
        dxdt = (1 / self.tau_e) * (-x)
        dydt = (1 / self.tau_f) * (-y + x)
        return torch.cat((dxdt, dydt))
    
    def steady_state(self):
        return torch.tensor([0., 0.])

class OU_noise:
    def __init__(self, tau_e=1.0, tau_f=1.0, sigma=1.0):
        # Define the model parameters (effective time constant, filtration time constant, and noise strength)
        self.tau_e = tau_e
        self.tau_f = tau_f
        self.sigma = sigma
        self.dim = 2

        # Define the jacobian
        self.J = torch.tensor([[-1 / self.tau_f, 0], [1 / self.tau_e, -1 / self.tau_e]])
    
    def noise_vector(self):
        # This function defines the noise vector for the OU noise process.
        return torch.tensor([self.sigma * math.sqrt(2 / self.tau_f), 0.])

    @dynm_fun
    def _dynamical_fun(self, t, vars):
        # This function defines the dynamics of the OU noise process.
        vars = vars.squeeze(0)  # Remove the extra dimension
        x = vars[0:1]
        y = vars[1:]
        dxdt = (1 / self.tau_f) * (-x)
        dydt = (1 / self.tau_e) * (-y + x)
        return torch.cat((dxdt, dydt))
    
    def steady_state(self):
        return torch.tensor([0., 0.])
    
class damped_harmonic_oscillator_noise:
    def __init__(self, zeta=0.5, omega0=1.0, temp=1.0):
        # Parameters: damping ratio (zeta), natural frequency (omega0), Temperature (temp, noise strength)
        self.zeta = zeta
        self.omega0 = omega0
        self.temp = temp
        self.dim = 2

        # Jacobian of the linearized system
        self.J = torch.tensor([[0, 1],
                               [-self.omega0**2, -2 * self.zeta * self.omega0]])

    def noise_vector(self):
        # Noise vector acts on velocity only
        return torch.tensor([0., math.sqrt(4 * self.zeta * self.omega0 * self.temp)])

    @dynm_fun
    def _dynamical_fun(self, t, vars):
        vars = vars.squeeze(0)  # Remove extra dimension
        x = vars[0:1]
        y = vars[1:]
        dxdt = y
        dydt = -2 * self.zeta * self.omega0 * y - (self.omega0**2) * x
        return torch.cat((dxdt, dydt))

    def steady_state(self):
        # Equilibrium at rest (zero displacement and velocity)
        return torch.tensor([0., 0.])