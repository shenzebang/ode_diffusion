import torch
from scipy import integrate
import numpy as np


## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5  # @param {'type': 'number'}


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                z=None,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that returns the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      atol: Tolerance of absolute errors.
      rtol: Tolerance of relative errors.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
      eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
                 * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
            print(f"t {time_steps[0]}, norm {torch.max(torch.sum(score ** 2, dim=(1, 2, 3)))}")
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol,
                              method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x