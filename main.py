import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from model import ScoreNet
from utils import marginal_prob_std_fn, diffusion_coeff_fn
from torchvision.utils import make_grid
from sampler import ode_sampler
from likelihood import ode_likelihood
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from nwgf import build_nwgf
from torch.nn import DataParallel
import os


# if torch.cuda.device_count() > 1:


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


def train_score_net_init(args):
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


def train_nwgf(args, device):
    # score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    # score_model = score_model.to(device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    score_0 = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    ckpt = torch.load('ckpt.pth', map_location=device)
    if not args.random_init:
        ## Load the pre-trained checkpoint from disk.
        score_model.load_state_dict(ckpt)
    score_0.load_state_dict(ckpt)

    n_epochs = args.n_epochs  # @param {'type':'integer'}
    ## size of a mini-batch
    batch_size = args.batch_size  # @param {'type':'integer'}
    ## learning rate
    lr = args.lr  # @param {'type':'number'}

    T = 1.
    exp_decay_fn = lambda t: args.exp_decay ** (T - t)
    nwgf_model = build_nwgf(
        score_net=score_model,
        score_0=score_0,
        diffusion_coeff_fn=diffusion_coeff_fn,
        time_length=T,
        exp_decay_fn=exp_decay_fn,
        atol=args.atol,
        rtol=args.rtol,
        score_0_t_0=args.score_0_t_0
    ).to(device)

    device_ids = [int(a) for a in args.device_ids.split(",")]
    if torch.cuda.device_count() > 1 and len(device_ids) > 1:
        nwgf_model = DataParallel(nwgf_model, device_ids).to(device)
    # nwgf_model = torch.nn.DataParallel(nwgf_model, device_ids=[4, 6], output_device=4).to(device)

    dataset = MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    model_path = os.path.join(args.dir, 'models')
    if not os.path.exists(model_path): os.makedirs(model_path)

    optimizer = Adam(nwgf_model.parameters(), lr=lr)
    tqdm_epoch = tqdm.trange(n_epochs)
    steps = 0
    time_0 = time.time()
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            steps += 1
            x = x.to(device)
            loss = torch.mean(nwgf_model(x))
            # print(loss.shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            if steps % 2 == 0:
                torch.save(score_model.state_dict(), os.path.join(model_path, f'nwgf_ckpt_{steps}.pth'))
            print(f"step {steps}, loss {loss.item()}, time {time.time() - time_0}")
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), os.path.join(model_path, 'nwgf_ckpt.pth'))


def test(args, device):
    # score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    # score_model = score_model.to(device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    step = args.i
    ## Load the pre-trained checkpoint from disk.

    model_path = os.path.join(args.dir, 'models')
    sample_path = os.path.join(args.dir, 'samples')
    if not os.path.exists(sample_path): os.makedirs(sample_path)

    if step > 0:
        ckpt = torch.load(os.path.join(model_path, f'nwgf_ckpt_{step}.pth'), map_location=device)
    else:
        ckpt = torch.load('ckpt.pth', map_location=device)
    score_model.load_state_dict(ckpt)

    sample_batch_size = 64  # @param {'type':'integer'}
    sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    ## Generate samples using the specified sampler.
    samples = sampler(score_model,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      sample_batch_size,
                      device=device,
                      eps=1e-3)

    print(torch.max(samples))

    ## Sample visualization.

    samples = samples.clamp(0.0, 1.0)

    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    if step > 0:
        plt.savefig(os.path.join(sample_path, f"sample_{step}.png"))
    else:
        plt.savefig(os.path.join(sample_path, "sample.png"))
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
    parser = argparse.ArgumentParser("diffusion")
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--random_init', action='store_true', help='Whether to load the pretrained score model')
    parser.add_argument('--i', type=int, default=50, help='test the performance at i th step')
    parser.add_argument('--n_epochs', type=int, default=50, help='max epochs of nwgf training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of nwgf training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of nwgf training')
    parser.add_argument('--exp_decay', type=float, default=1e-3, help='exponential decay rate of nwgf training')
    parser.add_argument('--score_0_t_0', type=float, default=1e-3, help='initial time of score_0')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--log', type=str, default='0')
    parser.add_argument('--atol', type=float, default=1e-4, help='absolute error tolerance when solving ODE')
    parser.add_argument('--rtol', type=float, default=1e-4, help='relative error tolerance when solving ODE')

    args = parser.parse_args()

    if not os.path.exists('ckpt.pth'):
        train_score_net_init(args)

    args.dir = os.path.join(f"log/exp{args.exp_decay}t_0{args.score_0_t_0}", args.log)
    if not os.path.exists(args.dir): os.makedirs(args.dir)

    device_ids = [int(a) for a in args.device_ids.split(",")]
    device = torch.device("cuda:" + str(min(device_ids)))

    print(args)
    if args.test:
        test(args, device)
    else:
        train_nwgf(args, device)

    # test_likelihood()
