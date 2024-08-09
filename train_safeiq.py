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
    n_mix_good = 400
    n_mix_bad = 1600
    n_bad = 100
    
    run_name = f'safeIQ(B={n_bad})'
    if (args.agent.pen_bad):
        run_name += '-pen'
    if (args.agent.cql):
        run_name += '-cql'
        
    wandb.init(project=f'test', settings=wandb.Settings(_disable_stats=True), \
            group=f'offline-{args.env.name}',
            job_type=run_name,
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

    LEARN_STEPS = int(env_args.learn_steps)

    agent = make_agent(env, args)

    c_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_C/mix_data.hdf5')
    c_dataset = load_dataset(c_data_path,num_trajectories=n_mix_good,seed=0)
    u_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_U/mix_data.hdf5')
    u_dataset = load_dataset(u_data_path,num_trajectories=n_mix_bad,seed=1)
    dataset = merge_dataset(c_dataset=c_dataset, u_dataset=u_dataset)
    mix_memory_replay = Memory(1, args.seed)
    mix_memory_replay.load_from_data(dataset)
    print(f'--> mix memory size: {mix_memory_replay.size()}')
    
    vio_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_U/bad_data.hdf5')
    vio_dataset = load_dataset(vio_data_path,num_trajectories=n_bad,seed=2)
    vio_dataset['is_constrained'] = np.zeros_like(vio_dataset['costs'],dtype=bool)
    bad_memory_replay = Memory(1, args.seed)
    bad_memory_replay.load_from_data(vio_dataset)
    print(f'--> Bad memory size: {bad_memory_replay.size()}')
    
    print(f'\n\nrun name = {run_name}')

    best_eval_returns = -np.inf
    learn_steps = 0

    # IQ-Learn Modification
    agent.update = types.MethodType(iq_update, agent)
    agent.update_critic = types.MethodType(update_critic, agent)
    agent.update_actor = types.MethodType(update_actor, agent)
    agent.update_actor_BC = types.MethodType(update_actor_BC, agent)
    agent.update_disc = types.MethodType(update_disc, agent)
    agent.pretrain_disc = types.MethodType(pretrain_disc, agent)
    
    disc_path = f'{hydra.utils.to_absolute_path("experts")}/{args.env.name}/Disc'
    agent.pretrain_disc(mix_memory_replay,bad_memory_replay,disc_path,f'disc_C({n_mix_good})_U({n_mix_bad})_B({n_bad})')
    agent.disc.eval()
    # Sample initial states from env
    for learn_steps in range(LEARN_STEPS+1):
        if (learn_steps%100 == 0):
            print(f'{learn_steps}/{LEARN_STEPS}'
            , end='\r')
        info = {}
        info.update(agent.update(mix_memory_replay,
                                 bad_memory_replay, learn_steps))
        
        if learn_steps % args.env.eval_interval == 0:
            eval_returns,eval_costs,cost_rate = eval_parallel(agent, eval_env, num_episodes=args.eval.eps,args = args)
            returns = np.mean(eval_returns)
            costs = np.mean(eval_costs)
            info.update({
                'eval/returns': round(returns, 3),
                'eval/costs': round(costs,3),
                'eval/cost_rate': round(cost_rate,3),
            })
            try:
                wandb.log(info,step=learn_steps)
            except:
                print(info)
            if returns > best_eval_returns:
                best_eval_returns = returns

def pretrain_disc(self, mix_buffer, bad_buffer, disc_path,file_name,total_step = 50000):
    os.makedirs(disc_path, exist_ok=True)
    file_path = f'{disc_path}/{file_name}'
    print(file_path)
    
    if (os.path.isfile(file_path)):
        print('load disc from ',file_path)
        self.disc.load_state_dict(torch.load(file_path, map_location=self.device))
    else:
        print('Pretrain disc')
        for itr in range(total_step+1):
            info = self.update_disc(mix_buffer,bad_buffer, itr)
            if (itr%1000 == 0):
                print(f'{itr}/{total_step} dif = {info["disc/bad"] - info["disc/mix"]:.3f}',info)
                if (info['disc/bad'] - info['disc/mix'] >0.2):
                    break
                
        torch.save(self.disc.state_dict(), file_path)
        print('save disc to ',file_path)
            
def save(agent, epoch, args, output_dir='results'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/weight')

def update_critic(self, mix_batch, bad_batch, step):
    args = self.args
    batch = get_concat_samples(mix_batch,bad_batch, args)
    obs,next_obs,action,reward,cost,done,is_constrained, is_bad = batch
    agent = self
    current_V = self.getV(obs)
    with torch.no_grad():
        next_V = self.get_targetV(next_obs).clip(min=-300,max=300)
        
    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent,agent.critic.Q1, current_Q1, current_V, next_V, batch,step)
        q2_loss, loss_dict2 = iq_loss(agent,agent.critic.Q2, current_Q2, current_V, next_V, batch,step)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch,step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()
    return loss_dict

def update_actor(self, obs ,exp_action,next_obs, step):
    action, log_prob, _ = self.actor.sample(obs)
    actor_Q = self.critic(obs, action)

    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

    # optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    info = {
        'actor_loss/target_entropy': round(self.target_entropy,1),
        'actor_loss/entropy': round(-log_prob.mean().item(),3)}

    return info

def update_actor_BC(self, obs, exp_action,next_obs,env_cost,is_constrained, step):
    with torch.no_grad():
        disc_reward,disc_prob = self.disc.get_weight(obs,exp_action,reward_weight=True)
        current_V = self.getV(obs).clip(min=-300,max=300)
        next_V = self.get_targetV(next_obs).clip(min=-300,max=300)
        Q_reward = current_V - self.gamma * next_V 
        adv_reward = (disc_reward - Q_reward.clip(min=-3, max=3)) * (1 - disc_prob)
        # adv_reward = dis_reward
        weight = torch.exp(adv_reward - adv_reward.max())
        weight = weight / weight.mean()
        weight[disc_reward > 0.55] = -1.0
    logp = self.actor.get_logp(obs,exp_action).clip(min=self.min_logp,max=self.max_logp)
    
    actor_loss = - (weight.detach() * logp).mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    
    loss_dict = {
        'actor_BC_loss/C_weight': round(weight[is_constrained.squeeze(-1)].mean().item(),3),
        'actor_BC_loss/U_weight': round(weight[~is_constrained.squeeze(-1)].mean().item(),3),
        
        'actor_BC_loss/C_prob': round(disc_prob[is_constrained.squeeze(-1)].mean().item(),3),
        'actor_BC_loss/U_prob': round(disc_prob[~is_constrained.squeeze(-1)].mean().item(),3),
        
        'actor_BC_loss/C_reward': round(disc_reward[is_constrained.squeeze(-1)].mean().item(),3),
        'actor_BC_loss/U_reward': round(disc_reward[~is_constrained.squeeze(-1)].mean().item(),3),
        
        'actor_BC_loss/C_Q_reward': round(Q_reward[is_constrained.squeeze(-1)].mean().item(),3),
        'actor_BC_loss/U_Q_reward': round(Q_reward[~is_constrained.squeeze(-1)].mean().item(),3),
        
        'actor_BC_loss/C_adv': round(adv_reward[is_constrained.squeeze(-1)].mean().item(),3),
        'actor_BC_loss/U_adv': round(adv_reward[~is_constrained.squeeze(-1)].mean().item(),3),
        
        'actor_BC_loss/C_logp': logp[is_constrained].mean().item(),
        'actor_BC_loss/U_logp': logp[~is_constrained].mean().item(),
        'actor_BC_loss/actor_loss': actor_loss.item(),
    }
    return loss_dict

def update_disc(self,mix_buffer,bad_buffer,step):
    mix_batch       = mix_buffer.get_samples(1024, self.device, noise=0.2)
    bad_batch       = bad_buffer.get_samples(1024, self.device, noise=0.2)
    mix_obs, _, mix_action, _,_, _,mix_is_constrained = mix_batch
    bad_obs, _, bad_action, _,_, _,bad_is_constrained = bad_batch
    
    mix_d = self.disc(mix_obs,mix_action)
    bad_d = self.disc(bad_obs,bad_action)
    loss_mix = -torch.log(1-mix_d).mean()
    loss_bad = -torch.log(bad_d).mean()

    loss = loss_mix + loss_bad
    
    self.disc_optimizer.zero_grad()
    loss.backward()
    self.disc_optimizer.step()

    if (step%1000 == 0):
        mix_batch       = mix_buffer.get_samples(5000, self.device)
        bad_batch       = bad_buffer.get_samples(5000, self.device)
        mix_obs, _, mix_action, _,_, _,mix_is_constrained = mix_batch
        bad_obs, _, bad_action, _,_, _,bad_is_constrained = bad_batch
        
        self.disc.eval()
        mix_d = self.disc(mix_obs,mix_action)
        bad_d = self.disc(bad_obs,bad_action)
        self.disc.train()
    
    info = {
        'disc/mix': round(mix_d.mean().item(),3),
        'disc/bad': round(bad_d.mean().item(),3),
        'disc/mix_good': round(mix_d[mix_is_constrained.squeeze(-1)].mean().item(),3),
        'disc/mix_bad': round(mix_d[~mix_is_constrained.squeeze(-1)].mean().item(),3),
    }
    return info

def iq_update(self, mix_buffer,bad_buffer, step):
    mix_batch       = mix_buffer.get_samples(self.batch_size, self.device)
    bad_batch       = bad_buffer.get_samples(self.batch_size, self.device)

    info = self.update_critic(mix_batch,bad_batch, step)
    # info.update(self.update_disc(mix_batch,bad_batch, step))
    
    obs, next_obs, action, env_reward,env_cost, done,is_constrained = mix_batch
    info.update(self.update_actor_BC(obs,action,next_obs,env_cost,is_constrained, step))

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    if (self.first_log):
        self.first_log = False
    return info

def iq_loss(agent,critic_Q, current_Q, current_v, next_v, batch,step):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward,env_cost, done,is_constrained, is_bad = batch

    loss_dict = {}
    v0 = agent.getV(obs).mean()

    y = (1 - done) * gamma * next_v
    with torch.no_grad():
        weight,_ = agent.disc.get_weight(obs,action,reward_weight=True)
        mix_weight = weight[~is_bad]
        bad_weight =weight[is_bad]
        
    mix_reward = (current_Q - y)[~is_bad]
    loss = - (mix_weight * mix_reward).mean()

    if (args.agent.pen_bad):
        bad_reward = (current_Q - y)[is_bad]
        if (agent.first_log):
            print('[critic] Pen bad')
        loss += - (bad_weight * bad_reward).mean()
        

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if args.method.loss == "value_expert":
        if (agent.first_log):
            print('[critic] value expert loss')
        value_loss = (current_v - y)[~is_bad].mean()
        loss += value_loss

    elif args.method.loss == "v0":
        if (agent.first_log):
            print('[critic] V0 loss')
        v0_loss = (1 - gamma) * v0
        loss += v0_loss

    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.type}')

    if (args.agent.cql):
        num_random = 25
        if (agent.first_log):
            print(f'[critic] CQL loss ({num_random} randoms)')
        cql_loss = agent.cqlV(obs[~is_bad.squeeze(1), ...], critic_Q, num_random) - current_Q[~is_bad].mean()
        loss += cql_loss
        loss_dict['critic_loss/cql_loss'] = round(cql_loss.item(),3)
        

    y = (1 - done) * gamma * next_v
    all_reward = (current_Q - y)
    chi2_loss = 0.5 * (all_reward**2).mean()
    loss += chi2_loss

    if (step%args.env.eval_interval == 0):
        with torch.no_grad():
            bad_reward = (current_Q - y)[is_bad]
            data_logp = agent.actor.get_logp(obs,action)
            pi_action, _, _ = agent.actor.sample(obs)
            pi_Q = agent.critic(obs, pi_action)
            pi_reward = (pi_Q - y).detach()[~is_bad]
            loss_dict['critic_loss/v0'] = round(v0.item(),3)
            loss_dict['critic_loss/mix_reward'] = round(mix_reward.mean().item(),3)
            loss_dict['critic_loss/bad_reward'] = round(bad_reward.mean().item(),3)
            loss_dict['critic_loss/pi_reward'] = round(pi_reward.mean().item(),3)
            loss_dict['critic_loss/mix_Q'] = round(current_Q[~is_bad].mean().item(),3)
            loss_dict['critic_loss/pi_Q'] = round(pi_Q[~is_bad].mean().item(),3)
            
            loss_dict['critic_loss/mix_logp'] = round(data_logp[~is_bad].mean().item(),3)
            loss_dict['critic_loss/bad_logp'] = round(data_logp[is_bad].mean().item(),3)
            loss_dict['critic_loss/mix_weight'] = round(mix_weight.mean().item(),3)
            loss_dict['critic_loss/bad_weight'] = round(bad_weight.mean().item(),3)
            loss_dict['True_value/C_weight'] = round(weight[is_constrained.squeeze(-1)].mean().item(),3)
            loss_dict['True_value/U_weight'] = round(weight[~is_constrained.squeeze(-1)].mean().item(),3)
            loss_dict['True_value/C_reward'] = round(all_reward[is_constrained].mean().item(),3)
            loss_dict['True_value/U_reward'] = round(all_reward[~is_constrained].mean().item(),3)
            loss_dict['True_value/C_Q'] = round(current_Q[is_constrained].mean().item(),3)
            loss_dict['True_value/U_Q'] = round(current_Q[~is_constrained].mean().item(),3)
            
    return loss, loss_dict


if __name__ == "__main__":
    main()
