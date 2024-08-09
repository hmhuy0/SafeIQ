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
from utils.utils import eval_mode, average_dicts, get_concat_samples, eval_parallel, soft_update, hard_update
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

def merge_dataset(dataset1,dataset2):
    dataset = {}
    dataset['states'] = np.concatenate([dataset1['states'],dataset2['states']],axis=0)
    dataset['actions'] = np.concatenate([dataset1['actions'],dataset2['actions']],axis=0)
    dataset['next_states'] = np.concatenate([dataset1['next_states'],dataset2['next_states']],axis=0)
    dataset['rewards'] = np.concatenate([dataset1['rewards'],dataset2['rewards']],axis=0)
    dataset['costs'] = np.concatenate([dataset1['costs'],dataset2['costs']],axis=0)
    dataset['dones'] = np.concatenate([dataset1['dones'],dataset2['dones']],axis=0)
    
    # random shuffle
    perm = np.random.permutation(len(dataset['states']))
    dataset['states'] = dataset['states'][perm]
    dataset['actions'] = dataset['actions'][perm]
    dataset['next_states'] = dataset['next_states'][perm]
    dataset['rewards'] = dataset['rewards'][perm]
    dataset['costs'] = dataset['costs'][perm]
    dataset['dones'] = dataset['dones'][perm]
    
    print('dataset shape:',dataset['states'].shape,dataset['actions'].shape,dataset['next_states'].shape,
          dataset['rewards'].shape,dataset['costs'].shape,dataset['dones'].shape,np.mean(dataset['costs']))
    return dataset

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    wandb.init(project=f'test', settings=wandb.Settings(_disable_stats=True), \
            group=f'offline-{args.env.name}',
            job_type=f'bc',
            name=f'{args.seed}', entity='hmhuy')

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    import safety_gymnasium
    env = make_env(args)
    eval_env = safety_gymnasium.vector.make(env_id=args.env.name, num_envs=args.eval.n_envs)

    # Seed envs
    eval_env.reset(seed=[i for i in range(args.eval.n_envs)])

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)

    agent = make_agent(env, args)
    # Load expert data
    c_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_C/mix_data.hdf5')
    c_dataset = load_dataset(c_data_path,num_trajectories=400,seed=0)
    u_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_U/mix_data.hdf5')
    u_dataset = load_dataset(u_data_path,num_trajectories=1600,seed=1)
    print()
    dataset = merge_dataset(c_dataset,u_dataset)
    
    expert_memory_replay = Memory(1, args.seed)
    expert_memory_replay.load_from_data(dataset)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    best_eval_returns = -np.inf
    learn_steps = 0

    # IQ-Learn Modification
    agent.update = types.MethodType(bc_update, agent)
    
    # Sample initial states from env
    for learn_steps in range(LEARN_STEPS+1):
        if (learn_steps%100 == 0):
            print(f'{learn_steps}/{LEARN_STEPS}'
            , end='\r')
        info = {}
        info.update(agent.update(expert_memory_replay, learn_steps))
        
        if learn_steps % args.env.eval_interval == 0:
            eval_returns,eval_costs,cost_rate = eval_parallel(agent, eval_env, num_episodes=100,args = args)
            returns = np.mean(eval_returns)
            costs = np.mean(eval_costs)
            info.update({
                'eval/returns': round(returns, 3),
                'eval/costs': round(costs,3),
                'eval/cost_rate': round(cost_rate,3),
            })
            try:
                wandb.log(info)
            except:
                print(info)
            if returns > best_eval_returns:
                best_eval_returns = returns


def save(agent, epoch, args, output_dir='results'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/weight')


def bc_update(self, expert_buffer, step):
    losses = {}
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
    obs,_,action = expert_batch[:3]
    logp = self.actor.get_logp(obs, action)
    bc_loss = -logp.mean()
    
    self.actor_optimizer.zero_grad()
    bc_loss.backward()
    self.actor_optimizer.step()
    losses['bc_loss'] = bc_loss.item()
    losses['bc_logp'] = logp.mean().item()
    return losses


if __name__ == "__main__":
    main()
