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

torch.set_num_threads(1)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
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
          f', Cost rate = {np.mean(trajs["costs"]):.3f}')
    return trajs

def merge_dataset(c_dataset,u_dataset):
    dataset = {}
    dataset['states'] = np.concatenate([c_dataset['states'],u_dataset['states']],axis=0)
    dataset['actions'] = np.concatenate([c_dataset['actions'],u_dataset['actions']],axis=0)
    dataset['next_states'] = np.concatenate([c_dataset['next_states'],u_dataset['next_states']],axis=0)
    dataset['rewards'] = np.concatenate([c_dataset['rewards'],u_dataset['rewards']],axis=0)
    dataset['costs'] = np.concatenate([c_dataset['costs'],u_dataset['costs']],axis=0)
    dataset['dones'] = np.concatenate([c_dataset['dones'],u_dataset['dones']],axis=0)
    dataset['is_constrained'] = np.concatenate([np.ones_like(c_dataset['costs'],dtype=bool),
                                                np.zeros_like(u_dataset['costs'],dtype=bool)],axis=0)
    
    # random shuffle
    perm = np.random.permutation(len(dataset['states']))
    dataset['states'] = dataset['states'][perm]
    dataset['actions'] = dataset['actions'][perm]
    dataset['next_states'] = dataset['next_states'][perm]
    dataset['rewards'] = dataset['rewards'][perm]
    dataset['costs'] = dataset['costs'][perm]
    dataset['dones'] = dataset['dones'][perm]
    dataset['is_constrained'] = dataset['is_constrained'][perm]
    
    return dataset

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    n_mix_good = args.expert.n_mix_good
    n_mix_bad = args.expert.n_mix_bad
    n_bad = args.expert.n_bad
    
    run_name = f'BC-safe'
        
    if (args.wandb_log):
        wandb.init(project=f'SafeIQ-{args.env.name}', settings=wandb.Settings(_disable_stats=True), \
                group=f'(mixG-mixB-B)=({n_mix_good}-{n_mix_bad}-{n_bad})',
                job_type=run_name,
                name=f'{args.seed}', entity='hmhuy')
    
    print(OmegaConf.to_yaml(cfg))
    
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

    LEARN_STEPS = int(env_args.learn_steps)

    agent = make_agent(env, args)

    # c_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_C/mix_data.hdf5')
    # c_dataset = load_dataset(c_data_path,num_trajectories=n_mix_good,seed=0)
    # u_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_U/mix_data.hdf5')
    # u_dataset = load_dataset(u_data_path,num_trajectories=n_mix_bad,seed=1)
    # dataset = merge_dataset(c_dataset=c_dataset, u_dataset=u_dataset)
    # del c_dataset
    # del u_dataset
    
    # mix_memory_replay = Memory(1, args.seed)
    # mix_memory_replay.load_from_data(dataset)
    # print(f'--> mix memory size: {mix_memory_replay.size()}')
    
    # vio_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_U/bad_data.hdf5')
    # vio_dataset = load_dataset(vio_data_path,num_trajectories=n_bad,seed=2)
    # vio_dataset['is_constrained'] = np.zeros_like(vio_dataset['costs'],dtype=bool)
    # bad_memory_replay = Memory(1, args.seed)
    # bad_memory_replay.load_from_data(vio_dataset)
    # print(f'--> Bad memory size: {bad_memory_replay.size()}')
    
    c_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_C/mix_data.hdf5')
    c_dataset = load_dataset(c_data_path,num_trajectories=n_mix_good,seed=0)
    c_dataset['is_constrained'] = np.ones_like(c_dataset['costs'],dtype=bool)
    mix_memory_replay = Memory(1, args.seed)
    mix_memory_replay.load_from_data(c_dataset)
    print(f'--> Good memory size: {mix_memory_replay.size()}')
    
    
    
    print(f'\n\nrun name = {run_name}')

    best_eval_returns = -np.inf
    learn_steps = 0



    # IQ-Learn Modification
    agent.update = types.MethodType(bc_update, agent)
    
    # Sample initial states from env
    for learn_steps in range(LEARN_STEPS+1):
        info = {}
        if (learn_steps % 5000 == 0):
            print(f'[{learn_steps}/{LEARN_STEPS}]')
        info.update(agent.update(mix_memory_replay, learn_steps))
        # info.update(agent.update(bad_memory_replay, learn_steps))

        if learn_steps % args.env.eval_interval == 0:
            eval_returns,eval_costs,cost_rate = eval_parallel(agent, eval_env, num_episodes=args.eval.eps,args = args)
            def top_10_percent_numpy(lst):
                np_array = np.array(lst)
                sorted_array = np.sort(np_array)[::-1]
                top_10_count = int(len(sorted_array) * 0.1)
                return sorted_array[:top_10_count]
            
            returns = np.mean(eval_returns)
            cvar_cost = np.mean(top_10_percent_numpy(eval_costs))
            costs = np.mean(eval_costs)
            print(f'[Eval {learn_steps/LEARN_STEPS*100:.1f}%] Returns: {returns:.2f}, Costs: {costs:.2f}, CVaR: {cvar_cost:.2f}, Cost rate: {cost_rate:.3f}')
            info.update({
                'eval/returns': round(returns, 3),
                'eval/costs': round(costs,3),
                'eval/cvar_cost': round(cvar_cost,3),
                'eval/cost_rate': round(cost_rate,3),
            })
            try:
                wandb.log(info,step=learn_steps)
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
    logp = self.actor.get_logp(obs, action).clip(min=-100,max=100)
    bc_loss = -logp.mean()
    
    self.actor_optimizer.zero_grad()
    bc_loss.backward()
    self.actor_optimizer.step()
    losses['bc_loss'] = bc_loss.item()
    losses['bc_logp'] = logp.mean().item()
    return losses


if __name__ == "__main__":
    main()
