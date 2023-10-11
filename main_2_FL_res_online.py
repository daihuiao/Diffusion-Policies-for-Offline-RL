# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy

import gym
import numpy as np
import os
import torch
import json

import d4rl
from tqdm import trange

from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter
import os
import wandb

from typing import Any, Dict, List, Optional, Tuple, Union
TensorBatch = List[torch.Tensor]

# 将此处的your_api_key替换为您的实际API密钥
os.environ["WANDB_API_KEY"] = "b4fdd4e5e894cba0eda9610de6f9f04b87a86453"
hyperparameters = {
    'walker2d-random-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                           'num_epochs': 2000, 'gn': 1.0, 'top_k': 1},
    'halfcheetah-medium-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                              'num_epochs': 2000, 'gn': 9.0, 'top_k': 1},
    'hopper-medium-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                         'num_epochs': 2000, 'gn': 9.0, 'top_k': 2},
    'walker2d-medium-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                           'num_epochs': 2000, 'gn': 1.0, 'top_k': 1},
    'halfcheetah-medium-replay-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no',
                                     'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0, 'top_k': 0},
    'hopper-medium-replay-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                                'num_epochs': 2000, 'gn': 4.0, 'top_k': 2},
    'walker2d-medium-replay-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                                  'num_epochs': 2000, 'gn': 4.0, 'top_k': 1},
    'halfcheetah-medium-expert-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no',
                                     'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0, 'top_k': 0},
    'hopper-medium-expert-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                                'num_epochs': 2000, 'gn': 5.0, 'top_k': 2},
    'walker2d-medium-expert-v2': {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                                  'num_epochs': 2000, 'gn': 5.0, 'top_k': 1},
    'antmaze-umaze-v0': {'lr': 3e-4, 'eta': 0.5, 'max_q_backup': False, 'reward_tune': 'cql_antmaze', 'eval_freq': 50,
                         'num_epochs': 1000, 'gn': 2.0, 'top_k': 2},
    'antmaze-umaze-diverse-v0': {'lr': 3e-4, 'eta': 2.0, 'max_q_backup': True, 'reward_tune': 'cql_antmaze',
                                 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0, 'top_k': 2},
    'antmaze-medium-play-v0': {'lr': 1e-3, 'eta': 2.0, 'max_q_backup': True, 'reward_tune': 'cql_antmaze',
                               'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0, 'top_k': 1},
    'antmaze-medium-diverse-v0': {'lr': 3e-4, 'eta': 3.0, 'max_q_backup': True, 'reward_tune': 'cql_antmaze',
                                  'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0, 'top_k': 1},
    'antmaze-large-play-v0': {'lr': 3e-4, 'eta': 4.5, 'max_q_backup': True, 'reward_tune': 'cql_antmaze',
                              'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0': {'lr': 3e-4, 'eta': 3.5, 'max_q_backup': True, 'reward_tune': 'cql_antmaze',
                                 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0, 'top_k': 1},
    'pen-human-v1': {'lr': 3e-5, 'eta': 0.15, 'max_q_backup': False, 'reward_tune': 'normalize', 'eval_freq': 50,
                     'num_epochs': 1000, 'gn': 7.0, 'top_k': 2},
    'pen-cloned-v1': {'lr': 3e-5, 'eta': 0.1, 'max_q_backup': False, 'reward_tune': 'normalize', 'eval_freq': 50,
                      'num_epochs': 1000, 'gn': 8.0, 'top_k': 2},
    'kitchen-complete-v0': {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                            'num_epochs': 250, 'gn': 9.0, 'top_k': 2},
    'kitchen-partial-v0': {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                           'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0': {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50,
                         'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}
class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]


def dataset_info(dataset):
    # 收集reward 信息
    rewards = []
    reward = 0
    trajectorys = 0
    lengths = []
    length = 0
    trajectory_lengths = []
    trajectory_length = 0
    for i in range(dataset["observations"].shape[0]):
        if not dataset["terminals"][i]:
            reward += dataset["rewards"][i]
            length += 1
            trajectory_length += 1
        elif dataset["terminals"][i]:
            reward += dataset["rewards"][i]
            rewards.append(reward)
            reward = 0
            length += 1
            lengths.append(length)
            length = 0
            trajectorys += 1
            trajectory_lengths.append(trajectory_length)
            trajectory_length = 0
        else:
            raise Exception
    sub_datasets = []
    num_agents = 10
    intervel = dataset["observations"].shape[0] // num_agents
    args.intervel = intervel
    for i in range(num_agents):
        sub_dataset = {}
        for key in dataset.keys():
            sub_dataset[key] = dataset[key][(i) * intervel:(i + 1) * intervel]
        sub_datasets.append(sub_dataset)
    print('=' * 50)
    print(f'{trajectorys} trajectories, {dataset["observations"].shape[0]} timesteps found')
    print(f'Average return: {np.mean(rewards):.2f}, std: {np.std(rewards):.2f}')
    try:
        print(f'Max return: {np.max(rewards):.2f}, min: {np.min(rewards):.2f}')
    except:
        print(f'Max return: nan, min: nan')

    print('=' * 50)
    return sub_datasets


def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    # data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner('Loaded buffer')
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        int(2e6),
        device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    if args.algo == 'ql':
        from agents.ql_diffusion_res_online import Diffusion_QL_res_online as Agent
        for i in range(args.num_agents):
            agent = Agent(state_dim=state_dim,
                          action_dim=action_dim,
                          max_action=max_action,
                          device=device,
                          discount=args.discount,
                          tau=args.tau,
                          max_q_backup=args.max_q_backup,
                          beta_schedule=args.beta_schedule,
                          n_timesteps=args.T,
                          eta=args.eta,
                          lr=args.lr,
                          lr_decay=args.lr_decay,
                          lr_maxt=args.num_epochs,
                          grad_norm=args.gn)
            agent.load_model(args.model_dir)
    elif args.algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr)

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)
    writer = SummaryWriter(output_dir)

    evaluations = []
    training_iters = 0
    max_timesteps = 0.2e6
    metric = 100.
    train_env = gym.make(args.env_name)
    train_env.seed(args.seed + 100)
    state, done = train_env.reset(), False

    utils.print_banner(f"Training Start", separator="*", num_star=90)
    for _ in trange(10000):
        action = agent.sample_action(np.array(state))
        next_state, reward, done, _ = train_env.step(action)
        replay_buffer.add_transition(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, done = train_env.reset(), False

    while (training_iters < max_timesteps) and (not early_stop):
        # iterations = int(args.eval_freq * args.num_steps_per_epoch)
        iterations = 100
        # loss_metric = agent.train(data_sampler,
        # collect_data(policy=agent, env=train_env,steps=iterations,replay_buffer=replay_buffer,device=device)
        for _ in range(iterations):
            action = agent.sample_action(np.array(state))
            next_state, reward, done, _ = train_env.step(action)
            replay_buffer.add_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                state, done = train_env.reset(), False
        # if train_only_actor:
        if False:
        # if True:
            loss_metric = agent.train_only_actor(replay_buffer,
                                      iterations=iterations,
                                      batch_size=args.batch_size,
                                      log_writer=writer)
        else:
            loss_metric = agent.train(replay_buffer,
                                      iterations=iterations,
                                      batch_size=args.batch_size,
                                      log_writer=writer)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Logging
        # logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.dump_tabular()
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        if training_iters % 10000 == 0:
            # Evaluation
            eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                                   eval_episodes=args.eval_episodes)
            evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                                np.mean(loss_metric['bc_loss']), np.mean(loss_metric['ql_loss']),
                                np.mean(loss_metric['actor_loss']), np.mean(loss_metric['critic_loss']),
                                curr_epoch])
            np.save(os.path.join(output_dir, "eval"), evaluations)
            logger.record_tabular('Average Episodic Reward', eval_res)
            logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
            logger.dump_tabular()
            wandb.log({"outereval/Average Episodic Reward": eval_res,
                       "outereval/Average Episodic N-Reward": eval_norm_res,
                       "outereval/Training_iters": training_iters})
            bc_loss = np.mean(loss_metric['bc_loss'])
            if args.early_stop:
                early_stop = stop_check(metric, bc_loss)

            metric = bc_loss

            if args.save_best_model:
                agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline
    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 2])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        bc_loss = scores[:, 4]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best normalized score avg': scores[where_k][0][2],
                    'best normalized score std': scores[where_k][0][3],
                    'best raw score avg': scores[where_k][0][0],
                    'best raw score std': scores[where_k][0][1]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

    # writer.close()

def collect_data(policy, env,  steps , replay_buffer, device):

    for _ in range(steps):
        state, done = env.reset(), False
        action = policy.sample_action(np.array(state))
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add_transition(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, done = env.reset(), False
    # return avg_reward, std_reward, avg_norm_score, std_norm_score
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score

# python main_FL_res.py --num_agents 3 --device 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_4', type=str)  # Experiment ID
    parser.add_argument('--device', default=1, type=int)  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="halfcheetah-medium-expert-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--model_dir", default="results/FL_res_fixed_2online|halfcheetah-medium-replay-v2|agent-10|T-5||ms-offline|k-0|0", type=str)  # OpenAI gym environment name
    # parser.add_argument("--env_name", default="walker2d-random-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)  # Logging directory
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--num_steps_per_epoch", default=200, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
    parser.add_argument("--num_agents", default=1, type=int)
    # parser.add_argument("--top_k", default=1, type=int)

    # parser.add_argument("--lr", default=3e-4, type=float)
    # parser.add_argument("--eta", default=1.0, type=float)
    # parser.add_argument("--max_q_backup", action='store_true')
    # parser.add_argument("--reward_tune", default='no', type=str)
    # parser.add_argument("--gn", default=-1.0, type=float)

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

    # Setup Logging
    file_name = f"FL_res_fixed|agent-{args.num_agents}|T-{args.T}|"
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    wandb.init(project="FDQL_to_online_", entity="aohuidai", mode="online",group=f"{args.env_name}", name=file_name, config=variant)
    # wandb.init(project="FDQL_to_online_", entity="aohuidai", mode="disabled",group=f"{args.env_name}", name=file_name, config=variant)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
