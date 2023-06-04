import copy
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb

# wandb.init(project="diffusion_demo",entity="aohuidai",mode="disabled")
# wandb.init(project="diffusion_demo",entity="aohuidai",mode="online")
# wandb.init(project="diffusion_demo", entity="aohuidai")

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/diffusion_demo')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
lr = 0.0003


class Dataset(object):
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def sample(self, batch_size):
        ind = torch.randint(0, self.data.shape[0], size=(batch_size,))
        return torch.clamp(torch.tensor(self.data[ind], dtype=torch.float32), min=-1, max=1).to(self.device)


# 定义均值和协方差矩阵
mean = [0.5, 0]
cov = [[0.01, 0], [0, 0.01]]

# 生成二维高斯分布
x, y = np.random.multivariate_normal(mean, cov, 50000).T
dataset0 = Dataset(np.random.multivariate_normal(mean, cov, 50000), device=device)
# 定义均值和协方差矩阵
mean = [-0.5, 0]
cov = [[0.01, 0], [0, 0.01]]
dataset1 = Dataset(np.random.multivariate_normal(mean, cov, 50000), device=device)
# 生成二维高斯分布
x2, y2 = np.random.multivariate_normal(mean, cov, 50000).T

# 绘制二维高斯分布的散点图
plt.scatter(x, y)
plt.scatter(x2, y2)
plt.show()

from agents.diffusion import Diffusion
from agents.model import MLP

model0 = MLP(state_dim=2, action_dim=2, device=device)
actor0 = Diffusion(state_dim=1, action_dim=2, model=model0, max_action=1.0,
                   beta_schedule='vp', n_timesteps=5,loss_type="l1" ).to(device)
actor0_optimizer = torch.optim.Adam(actor0.parameters(), lr=lr)

model1 = MLP(state_dim=2, action_dim=2, device=device)
actor1 = Diffusion(state_dim=1, action_dim=2, model=model1, max_action=1.0,
                   beta_schedule='vp', n_timesteps=5, ).to(device)
actor1_optimizer = torch.optim.Adam(actor1.parameters(), lr=lr)


def train(index, actor, actor_optimizer, dataset, epoch,iterations=1000):
    for _ in range(iterations):
        # if random.random() < 0.9:
        #     client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * index).to(device)
        # else:
        #     client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * -1).to(device)

        client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * index).to(device)
        states = torch.tensor(np.random.uniform(-1, 1, size=(128, 1), ), dtype=torch.float32).to(device)
        states = torch.cat([client_id, states], dim=1)
        actions = dataset.sample(128)
        actor_loss = actor.loss(actions, states)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        # if grad_norm > 0:
        #     actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        # wandb.log({f"loss{str(index)}/actor_loss": actor_loss})
        writer.add_scalar(f"loss{str(index)}/actor_loss", actor_loss, epoch*iterations+_)  # tensorboard
        actor_optimizer.step()
    return

global_parameters_actor = {}
for key, parameter in actor0.model.state_dict().items():
    global_parameters_actor[key] = parameter.clone()
actor1.model.load_state_dict(global_parameters_actor)

for i in range(100):
    train(0, actor0, actor0_optimizer, dataset0,epoch=i,iterations=100)
    train(1, actor1, actor1_optimizer, dataset1,epoch=i,iterations=100)
    for key, parameter in actor0.model.state_dict().items():
        global_parameters_actor[key] = parameter.clone()
    for key, parameter in actor1.model.state_dict().items():
        global_parameters_actor[key] += parameter.clone()
    for key, parameter in global_parameters_actor.items():
        global_parameters_actor[key] = parameter / 2.0
    actor0.model.load_state_dict(global_parameters_actor)
    actor1.model.load_state_dict(global_parameters_actor)
for i in range(500):
    index=0
    client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * index).to(device)
    states = torch.tensor(np.random.uniform(-1, 1, size=(128, 1), ), dtype=torch.float32).to(device)
    states = torch.cat([client_id, states], dim=1)
    actions = actor0.sample(states)
    x,y=actions[:,0].cpu().detach().numpy(),actions[:,1].cpu().detach().numpy()
    plt.scatter(x, y)
    index=1
    client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * index).to(device)
    states = torch.tensor(np.random.uniform(-1, 1, size=(128, 1), ), dtype=torch.float32).to(device)
    states = torch.cat([client_id, states], dim=1)
    actions = actor1.sample(states)
    x,y=actions[:,0].cpu().detach().numpy(),actions[:,1].cpu().detach().numpy()
    plt.scatter(x, y)

    index=-1
    client_id = torch.tensor(np.ones((128, 1), dtype=np.float32) * index).to(device)
    states = torch.tensor(np.random.uniform(-1, 1, size=(128, 1), ), dtype=torch.float32).to(device)
    states = torch.cat([client_id, states], dim=1)
    actions = actor1.sample(states)
    x,y=actions[:,0].cpu().detach().numpy(),actions[:,1].cpu().detach().numpy()
    plt.scatter(x, y)

    plt.show()
    haha=True

