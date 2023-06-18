import copy
import pickle
import random, datetime, time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import wandb
from torch.distributions import Normal
# wandb.init(project="diffusion_demo",entity="aohuidai",mode="disabled")
# wandb.init(project="diffusion_demo",entity="aohuidai",mode="online")
# wandb.init(project="diffusion_demo", entity="aohuidai")

from torch.utils.tensorboard import SummaryWriter

dt_obj = datetime.datetime.fromtimestamp(time.time())
formatted_time = dt_obj.strftime('%m-%d %H:%M')
import matplotlib.pyplot as plt
import  scienceplots
plt.style.use(['science','ieee'])

writer = SummaryWriter('runs/diffusion_demo')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.0003
# n_timesteps=5 #todo  一个对比实验
n_timesteps = 50


class Dataset(object):
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def sample(self, batch_size):
        ind = torch.randint(0, self.data.shape[0], size=(batch_size,))
        return torch.clamp(torch.tensor(self.data[ind], dtype=torch.float32), min=-1, max=1).to(self.device)

    def append(self, date):
        self.data = np.vstack((self.data, date))


# 定义均值和协方差矩阵
mean0 = [0.5, 0]
cov0 = [[0.01, 0], [0, 0.01]]
mean1 = [-0.5, 0]
cov1 = [[0.01, 0], [0, 0.01]]

# 生成二维高斯分布
dataset0 = Dataset(np.random.multivariate_normal(mean0, cov0, 50000), device=device)
dataset1 = Dataset(np.random.multivariate_normal(mean1, cov1, 50000), device=device)

# dataset0.append(np.random.multivariate_normal(mean1, cov1, 50000))
# dataset1.append(np.random.multivariate_normal(mean0, cov0, 50000))

# 生成二维高斯分布
x0, y0 = np.random.multivariate_normal(mean0, cov0, 50000).T
x1, y1 = np.random.multivariate_normal(mean1, cov1, 50000).T
# 绘制二维高斯分布的散点图
# plt.scatter(x0, y0,s=1)
plt.scatter(x0[:500], y0[:500],s=5)
# plt.scatter(x1, y1,s=1)
plt.scatter(x1[:500], y1[:500],s=5)
plt.axis('equal')
plt.show()

from agents.diffusion import Diffusion
from agents.model import MLP, MLP_res


class Actor(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
            need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


# model0 = MLP(state_dim=2, action_dim=2, device=device)
# model0 = MLP_res(state_dim=2, action_dim=2, device=device)
# actor0 = Diffusion(state_dim=1, action_dim=2, model=model0, max_action=1.0,
#                    beta_schedule='vp', n_timesteps=n_timesteps, loss_type="l2").to(device)
actor0 = Actor(state_dim=2, action_dim=2, hidden_dim=256, max_action=1.0).to(device)
actor0_optimizer = torch.optim.Adam(actor0.parameters(), lr=lr)

# model1 = MLP(state_dim=2, action_dim=2, device=device)
# model1 = MLP_res(state_dim=2, action_dim=2, device=device)
# actor1 = Diffusion(state_dim=1, action_dim=2, model=model1, max_action=1.0,
#                    beta_schedule='vp', n_timesteps=n_timesteps, loss_type="l2").to(device)
actor1 = Actor(state_dim=2, action_dim=2, hidden_dim=256, max_action=1.0).to(device)
actor1_optimizer = torch.optim.Adam(actor1.parameters(), lr=lr)


def train(index, actor, actor_optimizer, dataset, epoch, iterations=1000):
    for _ in range(iterations):
        if random.random() < 0.66:  # 0.9的概率用真实label
            client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * index).to(device)
        else:
            client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * -1).to(device)
        # 全都使用同一个label
        # client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * -1).to(device)
        states = torch.tensor(np.random.uniform(-1, 1, size=(128, 1), ), dtype=torch.float32).to(device)
        states = torch.cat([client_id, states], dim=1)
        actions = dataset.sample(128)

        recon, action_log_prob = actor(states, need_log_prob=True)

        recon_loss = F.mse_loss(recon, actions)
        actor_loss = recon_loss + (0.005 * action_log_prob).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        # if grad_norm > 0:
        #     actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        # wandb.log({f"loss{str(index)}/actor_loss": actor_loss})
        writer.add_scalar(f"loss{str(index)}/actor_loss", actor_loss, epoch * iterations + _)  # tensorboard
        actor_optimizer.step()
    return


global_parameters_actor = {}
for key, parameter in actor0.state_dict().items():
    global_parameters_actor[key] = parameter.clone()
actor1.load_state_dict(global_parameters_actor)


def plot_all():
    x_y = []
    index = 0
    client_id = torch.tensor(np.ones((500, 1), dtype=np.float32) * index).to(device)
    states = torch.tensor(np.random.uniform(-1, 1, size=(500, 1), ), dtype=torch.float32).to(device)
    states = torch.cat([client_id, states], dim=1)
    actions = actor0.act(states, device=device)
    x, y = actions[:, 0], actions[:, 1]
    x_y.append([x, y])
    plt.scatter(x, y,s=5)

    index = 1
    client_id = torch.tensor(np.ones((500, 1), dtype=np.float32) * index).to(device)
    states = torch.tensor(np.random.uniform(-1, 1, size=(500, 1), ), dtype=torch.float32).to(device)
    states = torch.cat([client_id, states], dim=1)
    actions = actor1.act(states, device=device)
    x, y = actions[:, 0], actions[:, 1]
    x_y.append([x, y])
    plt.scatter(x, y,s=5)

    index = -1
    client_id = torch.tensor(np.ones((500, 1), dtype=np.float32) * index).to(device)
    states = torch.tensor(np.random.uniform(-1, 1, size=(500, 1), ), dtype=torch.float32).to(device)
    states = torch.cat([client_id, states], dim=1)
    actions = actor0.act(states, device=device)
    x, y = actions[:, 0], actions[:, 1]
    x_y.append([x, y])
    plt.scatter(x, y,s=5)
    plt.axis('equal')
    plt.show()
    with open(f"x_y_gaussian_tanh_{formatted_time}.pkl", "ab") as f:
        pickle.dump(x_y, f)
    haha = True


def plot(actor0, i, index=0):
    client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * index).to(device)
    states = torch.tensor(np.random.uniform(-1, 1, size=(128, 1), ), dtype=torch.float32).to(device)
    states = torch.cat([client_id, states], dim=1)
    actions = actor0.sample(states)
    x, y = actions[:, 0].cpu().detach().numpy(), actions[:, 1].cpu().detach().numpy()
    plt.scatter(x, y)
    plt.title(f"agent:{index},epoch:{i},n_timesteps:{n_timesteps},iterations:{iterations}")
    plt.axis('equal')
    plt.show()


iterations = 100
for i in trange(1000):
    train(0, actor0, actor0_optimizer, dataset0, epoch=i, iterations=100)
    train(1, actor1, actor1_optimizer, dataset1, epoch=i, iterations=100)
    for key, parameter in actor0.state_dict().items():
        global_parameters_actor[key] = parameter.clone()
    for key, parameter in actor1.state_dict().items():
        global_parameters_actor[key] += parameter.clone()
    for key, parameter in global_parameters_actor.items():
        global_parameters_actor[key] = parameter / 2.0
    actor0.load_state_dict(global_parameters_actor)
    actor1.load_state_dict(global_parameters_actor)
    if (i + 1) % 100 == 0:
        # plot(actor0, i, index=-1)
        #
        # plot(actor0, i, index=0)
        # plot(actor1, i, index=1)
        plot_all()
