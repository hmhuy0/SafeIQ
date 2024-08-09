"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
import h5py

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import eval_mode, average_dicts, get_concat_samples, evaluate, soft_update, hard_update
from utils.logger import Logger
from iq import iq_loss

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg

def load_dataset(expert_location,
                 num_trajectories=None,seed=0):
    assert os.path.isfile(expert_location)
    
    hdf_trajs = h5py.File(expert_location, 'r')
    starts_timeout = np.where(np.array(hdf_trajs['timeouts'])>0)[0].tolist()
    starts_done = np.where(np.array(hdf_trajs['terminals'])>0)[0].tolist()
    starts = [-1]+starts_timeout+starts_done
    starts = list(dict.fromkeys(starts))
    starts.sort()
    
    rng = np.random.RandomState(seed)
    perm = np.arange(len(starts)-1)
    perm = rng.permutation(perm)
    if (num_trajectories):
        num_trajectories = min(num_trajectories,len(perm))
        idx = perm[:num_trajectories]
    else:
        idx = perm
    trajs = {}
    
    trajs['dones'] = [np.array(hdf_trajs['terminals'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    trajs['states'] = [np.array(hdf_trajs['observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    trajs['initial_states'] = np.array([hdf_trajs['observations'][starts[idx[i]]+1]
                        for i in range(len(idx))])
    trajs['next_states'] = [np.array(hdf_trajs['next_observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    trajs['actions'] = [np.array(hdf_trajs['actions'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    trajs['rewards'] = [hdf_trajs['rewards'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                            for i in range(len(idx))]
    trajs['costs'] = [hdf_trajs['costs'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                            for i in range(len(idx))]
    
    reward_arr = [np.sum(trajs['rewards'][i]) for i in range(len(trajs['rewards']))]
    
    trajs['dones'] = np.concatenate(trajs['dones'],axis=0)
    trajs['states'] = np.concatenate(trajs['states'],axis=0)
    trajs['actions'] = np.concatenate(trajs['actions'],axis=0)
    trajs['next_states'] = np.concatenate(trajs['next_states'],axis=0)
    
    trajs['rewards'] = np.concatenate(trajs['rewards'],axis=0)
    trajs['costs'] = np.concatenate(trajs['costs'],axis=0)
    
    print(f'expert: {expert_location}, {len(idx)}/{len(perm)} trajectories')
    print('dataset shape:',trajs['states'].shape,trajs['actions'].shape,trajs['next_states'].shape,
          trajs['rewards'].shape)
    print(f'Return = {np.mean(reward_arr):.2f} +- {np.std(reward_arr):.2f}'+
          f', Cost rate = {np.mean(trajs["costs"]):.2f}')
    return trajs


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    # wandb.init(project=args.project_name, entity='iq-learn',
    #            sync_tensorboard=True, reinit=True, config=args)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    env = make_env(args)
    eval_env = make_env(args)

    # Seed envs
    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 10)

    LEARN_STEPS = int(env_args.learn_steps)

    agent = make_agent(env, args)
    # Load expert data
    data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_C/mix_data.hdf5')
    dataset = load_dataset(data_path,num_trajectories=300)
    expert_memory_replay = Memory(1, args.seed)
    expert_memory_replay.load_from_data(dataset)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    best_eval_returns = -np.inf
    learn_steps = 0

    # IQ-Learn Modification
    agent.iq_update = types.MethodType(iq_update, agent)
    agent.update_critic = types.MethodType(update_critic, agent)
    agent.update_actor = types.MethodType(update_actor, agent)
    
    # Sample initial states from env
    for learn_steps in range(LEARN_STEPS+1):
        if (learn_steps%100 == 0):
            print(f'{learn_steps}/{LEARN_STEPS}'
            , end='\r')
        info = {}
        info.update(agent.iq_update(expert_memory_replay, learn_steps))
        
        if learn_steps % args.env.eval_interval == 0:
            eval_returns,eval_costs, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
            returns = np.mean(eval_returns)
            costs = np.mean(eval_costs)
            timesteps = np.mean(eval_timesteps)
            info.update({
                'eval/returns': returns,
                'eval/costs': costs,
                'eval/timesteps': timesteps
            })
            print(info)
            if returns > best_eval_returns:
                best_eval_returns = returns


def save(agent, epoch, args, output_dir='results'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/weight')

def update_critic(self, policy_batch, expert_batch, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs).clip(min=-100,max=100)
    else:
        next_V = self.getV(next_obs).clip(min=-100,max=100)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
        q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()
    return loss_dict

def update_actor(self, obs, step):
    action, log_prob, _ = self.actor.sample(obs)
    actor_Q = self.critic(obs, action)

    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

    # optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    losses = {
        'loss/actor': actor_loss.item(),
        'actor_loss/target_entropy': self.target_entropy,
        'actor_loss/entropy': -log_prob.mean().item()}

    # self.actor.log(logger, step)
    if self.learn_temp:
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                        (-log_prob - self.target_entropy).detach()).mean()

        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        losses.update({
            'alpha_loss/loss': alpha_loss.item(),
            'alpha_loss/value': self.alpha.item(),
        })
    return losses

def iq_update(self, expert_buffer, step):
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.update_critic(expert_batch, step)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            if self.args.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor(obs, step)

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


if __name__ == "__main__":
    main()
