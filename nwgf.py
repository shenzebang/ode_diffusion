import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def sample_gaussian_like(y):
    return torch.randn_like(y)


class ODEDiffusioin(nn.Module):
    def __init__(self, score_net, diffusion_coeff_fn, exp_decay=1.0):
        super(ODEDiffusioin, self).__init__()
        self.score_net = score_net
        self.exp_decay = exp_decay
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))
        self.diffusion_coeff_fn = diffusion_coeff_fn

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        x = states[0]
        score = states[1]
        t = torch.tensor(t).type_as(x).to(x)
        eps = 0
        _t = torch.ones(x.shape[0], device=x.device) * (t * (1. - eps) + eps)

        batchsize, n_channels, dim_1, dim_2 = x.shape

        # Sample and fix the noise.
        if self._e is None:
            self._e = sample_gaussian_like(x)

        diffusion_weight = self.diffusion_coeff_fn(t)

        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            dx = - diffusion_weight ** 2 * self.score_net(x, _t) / 2
            divergence = self.divergence_fn(dx, x, e=self._e).view(batchsize, 1)
            dscore_1 = - torch.autograd.grad(torch.sum(divergence), x, create_graph=True)[0]
            dscore_2 = - torch.autograd.grad(torch.sum(dx * score.detach()), x, create_graph=True)[0]
            dscore = dscore_1 + dscore_2
            dwgf_reg = dx + diffusion_weight ** 2 * score / 2
            dwgf_reg = self.exp_decay ** t * torch.sum(dwgf_reg ** 2) / x.shape[0]

        # increment num evals
        self._num_evals += 1
        # print(f"t {t}, R {states[-1]}")
        # print(f"t {t}, dR {dwgf_reg}")
        # print(f"t {t}, score {torch.norm(states[1])/ x.shape[0]}")
        # print(f"t {t}, dscore_1 {torch.norm(dscore_1)/ x.shape[0]}")
        # print(f"t {t}, dscore_2 {torch.norm(dscore_2)/ x.shape[0]}")
        # print(f"t {t}, dx {torch.norm(dx)}")
        # print(f"t {t}, divergence {torch.norm(divergence)/x.shape[0]}")
        return tuple([dx, dscore, dwgf_reg])


class NWGFDiffusion(nn.Module):
    def __init__(self, ode_diffusion, score_0, T=1.0, solver='dopri5', atol=1e-3, rtol=1e-3):
        super(NWGFDiffusion, self).__init__()
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.T = T
        self.ode_diffusion = ode_diffusion
        self.score_0 = score_0

    def forward(self, x_0):
        eps = 1e-5
        integration_times = torch.tensor([0.0+eps, self.T]).to(x_0)

        t_0 = torch.zeros(x_0.shape[0], device=x_0.device) * (1. - eps) + eps



        score_0 = self.score_0(x_0, t_0).detach()
        wgf_reg_0 = torch.zeros([1]).to(x_0)

        self.ode_diffusion.before_odeint()

        state_t = odeint(
            self.ode_diffusion,
            (x_0, score_0, wgf_reg_0),
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        x_t, score_t, wgf_reg_t = state_t

        return wgf_reg_t

    def num_evals(self):
        return self.ode_diffusion._num_evals.item()


def build_nwgf(score_net, score_0, diffusion_coeff_fn, time_length=1.0, exp_decay=1.0):
    ode_diffusion = ODEDiffusioin(
        score_net=score_net,
        diffusion_coeff_fn=diffusion_coeff_fn,
        exp_decay=exp_decay
    )
    model = NWGFDiffusion(
        ode_diffusion=ode_diffusion,
        score_0=score_0,
        T=time_length
    )

    return model
