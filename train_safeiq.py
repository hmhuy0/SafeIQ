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
    
    run_name = f'final-our({args.method.loss})'
    if (args.agent.pen_bad):
        run_name += '-penB'
        
    if (args.wandb_log):
        wandb.login(key="2b82ac1a4a3010132ac1ba34948e6d705cd6343b")
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

    c_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_C/mix_data.hdf5')
    c_dataset = load_dataset(c_data_path,num_trajectories=n_mix_good,seed=0)
    u_data_path = hydra.utils.to_absolute_path(f'experts/{args.env.name}/collect_U/mix_data.hdf5')
    u_dataset = load_dataset(u_data_path,num_trajectories=n_mix_bad,seed=1)
    dataset = merge_dataset(c_dataset=c_dataset, u_dataset=u_dataset)
    del c_dataset
    del u_dataset
    
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
    agent.update_actor_BC = types.MethodType(update_actor_BC, agent)
    agent.update_disc = types.MethodType(update_disc, agent)
    agent.pretrain_disc = types.MethodType(pretrain_disc, agent)
    
    disc_path = f'{hydra.utils.to_absolute_path("experts")}/{args.env.name}/Disc'
    agent.pretrain_disc(mix_memory_replay,bad_memory_replay,disc_path,f'state_disc_C({n_mix_good})_U({n_mix_bad})_B({n_bad})')
    agent.disc_mix.eval()
    agent.disc_bad.eval()
    
    for learn_steps in range(LEARN_STEPS+1):
        info = {}
        if (learn_steps % 1000 == 0):
            print(f'[{learn_steps}/{LEARN_STEPS}]')
        info.update(agent.update(mix_memory_replay,
                                 bad_memory_replay, learn_steps))
        
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

def pretrain_disc(self, mix_buffer, bad_buffer, disc_path,file_name,total_step = 50000):
    os.makedirs(disc_path, exist_ok=True)
    print(disc_path)
    print(file_name)
    
    mix_path = f'{disc_path}/mix_{file_name}'
    bad_path = f'{disc_path}/bad_{file_name}'
    
    if (os.path.isfile(mix_path)):
        print('load mix disc from ',mix_path)
        self.disc_mix.load_state_dict(torch.load(mix_path, map_location=self.device))
        print('load bad disc from ',bad_path)
        self.disc_bad.load_state_dict(torch.load(bad_path, map_location=self.device))
    else:
        print('Pretrain disc mix')
        print(mix_path)
        for itr in range(total_step+1):
            info = self.update_disc(mix_buffer,bad_buffer,mix_path, itr)
            if (itr%1000 == 0):
                print(f'{itr}/{total_step} dif = {abs(info["disc/bad"] - info["disc/mix"]):.3f}',info)
                if (abs(info['disc/bad'] - info['disc/mix']) >0.2):
                    break
        torch.save(self.disc_mix.state_dict(), mix_path)
        print('save disc mix to ',mix_path)      
        print('-'*20)       
   
        print('Pretrain disc bad')
        print(bad_path)
        for itr in range(total_step+1):
            info = self.update_disc(mix_buffer,bad_buffer,bad_path, itr)
            if (itr%1000 == 0):
                print(f'{itr}/{total_step} dif = {abs(info["disc/bad"] - info["disc/mix"]):.3f}',info)
                if (abs(info['disc/bad'] - info['disc/mix']) >0.2):
                    break
                
        torch.save(self.disc_bad.state_dict(), bad_path)
        print('save disc bad to ',bad_path)      
        print('-'*20)          

def update_disc(self,mix_buffer,bad_buffer,file_path,step):
    if ('mix' in file_path.split('/')[-1]):
        self.disc = self.disc_mix
        self.disc_optimizer = self.disc_mix_optimizer
    else:
        self.disc = self.disc_bad
        self.disc_optimizer = self.disc_bad_optimizer
        
    mix_batch       = mix_buffer.get_samples(1024, self.device, noise=0.0)
    bad_batch       = bad_buffer.get_samples(1024, self.device, noise=0.0)
    mix_obs, mix_next_obs, mix_action, _,_, _,mix_is_constrained = mix_batch
    bad_obs, bad_next_obs, bad_action, _,_, _,bad_is_constrained = bad_batch

    mix_d = self.disc(mix_obs)
    bad_d = self.disc(bad_obs)
    
    if ('bad' in file_path.split('/')[-1]):
        loss_bad = - torch.log(bad_d).mean()
        loss_mix = - torch.log(1-mix_d).mean()
        
        # if ('overlap' in file_path.split('/')[-1]):
        #     loss_mix = -1.5 * torch.log(1-mix_d).mean() + torch.log(1-bad_d).mean()
        # else:
        #     loss_mix = - torch.log(1-mix_d).mean()
    else:
        loss_bad = - torch.log(1-bad_d).mean()
        loss_mix = - torch.log(mix_d).mean()
        
        # if ('overlap' in file_path.split('/')[-1]):
        #     loss_mix = -1.5 * torch.log(mix_d).mean() + torch.log(bad_d).mean()
        # else:
        #     loss_mix = - torch.log(mix_d).mean()
        
    
    # if ('overlap' in file_path.split('/')[-1] and 
    #     'bad' in file_path.split('/')[-1]):
    #     loss_mix = -1.5 * torch.log(1-mix_d).mean() + torch.log(1-bad_d).mean()
    # else:
    #     loss_mix = - torch.log(1-mix_d).mean()
        
    # if ('overlap' in file_path.split('/')[-1] and 
    #     'good' in file_path.split('/')[-1]):
    #     loss_bad = -1.5 * torch.log(bad_d).mean() + torch.log(mix_d).mean()
    # else:
    #     loss_bad = - torch.log(bad_d).mean()
        
    loss = loss_mix + loss_bad
    
    
    self.disc_optimizer.zero_grad()
    loss.backward()
    self.disc_optimizer.step()
    info = {}
    
    if (step%1000 == 0):
        mix_batch       = mix_buffer.get_samples(5000, self.device)
        bad_batch       = bad_buffer.get_samples(5000, self.device)
        self.disc.eval()
        mix_obs, mix_next_obs, mix_action, _,_, _,mix_is_constrained = mix_batch
        bad_obs, bad_next_obs, bad_action, _,_, _,bad_is_constrained = bad_batch

        mix_d = self.disc(mix_obs)
        bad_d = self.disc(bad_obs)

        self.disc.train()
    
        info = {
            'disc/mix': round(mix_d.mean().item(),3),
            'disc/bad': round(bad_d.mean().item(),3),
            'disc/mix_good': round(mix_d[mix_is_constrained.squeeze(-1)].mean().item(),3),
            'disc/mix_bad': round(mix_d[~mix_is_constrained.squeeze(-1)].mean().item(),3),
        }
    return info
            
def save(agent, epoch, args, output_dir='results'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/weight')

def update_value(self, obs, action, next_obs, step):
    def iql_loss(pred, target, expectile=0.7):
        err = target - pred
        weight = torch.abs(expectile - (err < 0).float())
        return (weight * torch.square(err)).mean()
    
    args = self.args
    cur_V = self.getV(obs)
    with torch.no_grad():
        cur_Q = self.critic_target(obs, action)
        
    loss = iql_loss(cur_V,cur_Q,0.7)
    
    self.value_optimizer.zero_grad()
    loss.backward()
    self.value_optimizer.step()
    
    infos = {}
    if (step%args.env.eval_interval == 0):
        infos = {
            'value/value_loss': round(loss.item(),3),
            'value/V': round(cur_V.mean().item(),3),
            'value/Q': round(cur_Q.mean().item(),3),
        }

    return infos
    
def update_critic(self, mix_batch, bad_batch, step):
    args = self.args
    batch = get_concat_samples(mix_batch,bad_batch, args)
    obs,next_obs,action,reward,cost,done,is_constrained, is_bad = batch
    agent = self
    
    # update V
    infos = update_value(self, obs, action, next_obs, step)
    
    # update Q    
    with torch.no_grad():
        next_V = self.getV(next_obs).clip(min=-self.max_v,max=self.max_v)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent,agent.critic.Q1, current_Q1, next_V, batch,step)
        q2_loss, loss_dict2 = iq_loss(agent,agent.critic.Q2, current_Q2, next_V, batch,step)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        raise NotImplementedError
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch,step)
    infos.update(loss_dict)
    
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    
    return infos

def update_actor_BC(self, obs, exp_action,next_obs,
                    done,env_cost,is_constrained,step):
    with torch.no_grad():
        Qs = self.critic_target(obs,exp_action).clip(min=-self.max_v,max=self.max_v)
        current_V = self.getV(obs).clip(min=-self.max_v,max=self.max_v)
        
        cur_weight = self.get_weight(obs) + self.gamma * self.get_weight(next_obs)
        disc_weight = (1/(cur_weight**2)).clip(min=0,max=100)

        Q_adv = (Qs - current_V).clip(max=10)
        adv = Q_adv
        
        adv_weight = torch.exp(adv - adv.max())
        adv_weight = adv_weight / adv_weight.mean()
        
        weight = (adv_weight*disc_weight).clip(max=100)
        
    logp = self.actor.get_logp(obs,exp_action).clip(min=self.min_logp,max=self.max_logp)
    
    actor_loss = - (weight.detach() * logp).mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    
    loss_dict = {}
    if (step%self.args.env.eval_interval == 0):
        loss_dict = {
            'actor_BC_loss/C_weight': round(weight[is_constrained.squeeze(-1)].mean().item(),3),
            'actor_BC_loss/U_weight': round(weight[~is_constrained.squeeze(-1)].mean().item(),3),
            
            'histogram/C_weight': wandb.Histogram(weight[is_constrained.squeeze(-1)].cpu().numpy()),
            'histogram/U_weight': wandb.Histogram(weight[~is_constrained.squeeze(-1)].cpu().numpy()),
            
            'actor_BC_loss/C_disc_weight': round(disc_weight[is_constrained.squeeze(-1)].mean().item(),3),
            'actor_BC_loss/U_disc_weight': round(disc_weight[~is_constrained.squeeze(-1)].mean().item(),3),

            'histogram/C_disc_weight': wandb.Histogram(disc_weight[is_constrained.squeeze(-1)].cpu().numpy()),
            'histogram/U_disc_weight': wandb.Histogram(disc_weight[~is_constrained.squeeze(-1)].cpu().numpy()),

            'actor_BC_loss/C_adv': round(Q_adv[is_constrained.squeeze(-1)].mean().item(),3),
            'actor_BC_loss/U_adv': round(Q_adv[~is_constrained.squeeze(-1)].mean().item(),3),
            
            'actor_BC_loss/C_logp': logp[is_constrained].mean().item(),
            'actor_BC_loss/U_logp': logp[~is_constrained].mean().item(),
            
            'actor_BC_loss/actor_loss': actor_loss.item(),
        }
    return loss_dict

def iq_update(self, mix_buffer,bad_buffer, step):
    mix_batch       = mix_buffer.get_samples(self.batch_size, self.device)
    bad_batch       = bad_buffer.get_samples(self.batch_size, self.device)

    info = self.update_critic(mix_batch,bad_batch, step)
    
    mix_batch       = mix_buffer.get_samples(self.batch_size*2, self.device)
    obs, next_obs, action, env_reward,env_cost, done,is_constrained = mix_batch
    info.update(self.update_actor_BC(obs,action,next_obs,done,env_cost,is_constrained, step))

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    if (self.first_log):
        self.first_log = False
    return info

def iq_loss(agent,critic_Q, current_Q, next_v, batch,step):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward,env_cost, done,is_constrained, is_bad = batch

    loss_dict = {}

    y = (1 - done) * gamma * next_v
    with torch.no_grad():
        cur_weight = agent.get_weight(obs,is_bad)
        next_weight = agent.get_weight(next_obs,is_bad)
        weight = (cur_weight + gamma * next_weight).clip(min=-agent.reward_factor,max=agent.reward_factor)
        
    reward = (current_Q - y)
    if (not args.agent.pen_bad):
        reward = reward[~is_bad]
        weight = weight[~is_bad]
    else:
        if (agent.first_log):
            print('[critic] penalize bad samples')
    reward_loss = -(weight * reward).mean()

    y = (1 - done) * gamma * next_v
    all_reward = (current_Q - y)
    chi2_loss = 0.5 * (all_reward**2).mean()
    
    loss = reward_loss + chi2_loss
    
    if args.method.loss == "v0":
        if (agent.first_log):
            print('[critic] v0 value loss')
        pi_action,_,_ = agent.actor.sample(obs)
        v0 = agent.critic(obs, pi_action).mean()
        v0_loss = (1 - gamma) * v0
        loss += v0_loss
    elif args.method.loss == "no":
        if (agent.first_log):
            print('[critic] no value loss')
    else:
        raise NotImplementedError

    if (step%args.env.eval_interval == 0):
        with torch.no_grad():
            v0 = agent.getV(obs).mean()
            bad_reward = (current_Q - y)[is_bad]
            data_logp = agent.actor.get_logp(obs,action)
            pi_action, _, _ = agent.actor.sample(obs)
            pi_Q = agent.critic(obs, pi_action)
            pi_reward = (pi_Q - y).detach()[~is_bad]
            loss_dict['critic_loss/v0'] = round(v0.item(),3)
            loss_dict['critic_loss/mix_reward'] = round(reward[~is_bad].mean().item(),3)
            loss_dict['critic_loss/bad_reward'] = round(reward[is_bad].mean().item(),3)
            loss_dict['critic_loss/pi_reward'] = round(pi_reward.mean().item(),3)
            loss_dict['critic_loss/mix_Q'] = round(current_Q[~is_bad].mean().item(),3)
            loss_dict['critic_loss/pi_Q'] = round(pi_Q[~is_bad].mean().item(),3)
            loss_dict['critic_loss/mix_logp'] = round(data_logp[~is_bad].mean().item(),3)
            loss_dict['critic_loss/bad_logp'] = round(data_logp[is_bad].mean().item(),3)
            loss_dict['critic_loss/mix_weight'] = round(weight[~is_bad].mean().item(),3)
            loss_dict['critic_loss/bad_weight'] = round(weight[is_bad].mean().item(),3)
            
            loss_dict['True_value/C_weight'] = round(weight[is_constrained.squeeze(-1)].mean().item(),3)
            loss_dict['True_value/U_weight'] = round(weight[~is_constrained.squeeze(-1)].mean().item(),3)
            loss_dict['True_value/C_reward'] = round(all_reward[is_constrained].mean().item(),3)
            loss_dict['True_value/U_reward'] = round(all_reward[~is_constrained].mean().item(),3)
            loss_dict['True_value/C_Q'] = round(current_Q[is_constrained].mean().item(),3)
            loss_dict['True_value/U_Q'] = round(current_Q[~is_constrained].mean().item(),3)
            
    return loss, loss_dict


if __name__ == "__main__":
    main()