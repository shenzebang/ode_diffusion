import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from model import ScoreNet
from utils import marginal_prob_std_fn, diffusion_coeff_fn, device
from torchvision.utils import make_grid
from sampler import ode_sampler
from likelihood import ode_likelihood
import matplotlib.pyplot as plt
import numpy as np

from nwgf import build_nwgf

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))
    return loss


def train_score_net_init():
    # score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    # score_model = score_model.to(device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)

    n_epochs = 50  # @param {'type':'integer'}
    ## size of a mini-batch
    batch_size = 32  # @param {'type':'integer'}
    ## learning rate
    lr = 1e-4  # @param {'type':'number'}

    dataset = MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(score_model.parameters(), lr=lr)
    tqdm_epoch = tqdm.trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)
            # x = x / 255.
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), 'ckpt.pth')

def train_nwgf():
    # score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    # score_model = score_model.to(device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    score_0 = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    ## Load the pre-trained checkpoint from disk.
    ckpt = torch.load('ckpt.pth', map_location=device)
    score_model.load_state_dict(ckpt)
    score_0.load_state_dict(ckpt)

    n_epochs = 50  # @param {'type':'integer'}
    ## size of a mini-batch
    batch_size = 64  # @param {'type':'integer'}
    ## learning rate
    lr = 1e-4  # @param {'type':'number'}

    T = 1
    exp_decay = 1./(25.**4)
    nwgf_model = build_nwgf(
        score_net=score_model,
        score_0=score_0,
        diffusion_coeff_fn=diffusion_coeff_fn,
        time_length=T,
        exp_decay=exp_decay
    )

    dataset = MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    optimizer = Adam(nwgf_model.parameters(), lr=lr)
    tqdm_epoch = tqdm.trange(n_epochs)
    steps = 0
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            steps += 1
            x = x.to(device)
            # x = x / 255.
            loss = nwgf_model(x)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            # if steps % 50 == 0:
            torch.save(score_model.state_dict(), f'nwgf_ckpt_{steps}.pth')
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), 'nwgf_ckpt.pth')

def test():
    # score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    # score_model = score_model.to(device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    step = 20
    ## Load the pre-trained checkpoint from disk.
    ckpt = torch.load(f'nwgf_ckpt_{step}.pth', map_location=device)
    # ckpt = torch.load('ckpt.pth', map_location=device)
    score_model.load_state_dict(ckpt)

    sample_batch_size = 64  # @param {'type':'integer'}
    sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    ## Generate samples using the specified sampler.
    samples = sampler(score_model,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      sample_batch_size,
                      device=device)

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)

    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.savefig(f"sample_{step}.png")
    # plt.savefig("sample.png")
    plt.show()






def test_likelihood():
    # score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    # score_model = score_model.to(device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)

    batch_size = 32  # @param {'type':'integer'}

    dataset = MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ckpt = torch.load('ckpt.pth', map_location=device)
    score_model.load_state_dict(ckpt)

    all_bpds = 0.
    all_items = 0
    try:
        tqdm_data = tqdm.tqdm(data_loader)
        for x, _ in tqdm_data:
            x = x.to(device)
            # uniform dequantization
            x = (x * 255. + torch.rand_like(x)) / 256.
            _, bpd = ode_likelihood(x, score_model, marginal_prob_std_fn,
                                    diffusion_coeff_fn,
                                    x.shape[0], device=device, eps=1e-5)
            all_bpds += bpd.sum()
            all_items += bpd.shape[0]
            tqdm_data.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))

    except KeyboardInterrupt:
        # Remove the error message when interuptted by keyboard or GUI.
        pass



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # train_score_net_init()
    # train_nwgf()
    test()
    # test_likelihood()
