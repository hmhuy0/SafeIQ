import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def eval_parallel(actor, env, num_episodes, args):
    total_returns = []
    total_costs = []
    total_steps = 0
    while len(total_returns) < num_episodes:
        print(f'[Eval] {len(total_returns)}/{num_episodes}'
            , end='\r')
        state,_ = env.reset()
        done = [False]
        total_return = np.array([0.0 for _ in range(args.eval.n_envs)])
        total_cost = np.array([0.0 for _ in range(args.eval.n_envs)])
        with eval_mode(actor):
            while not np.max(done):
                action = actor.get_batch_action(state, sample=False)
                next_state, reward,cost, done,trunc, info = env.step(action)
                state = next_state
                total_return += reward
                total_cost += cost
                total_steps += len(cost)
                if (np.max(done) or np.max(trunc)):
                    for idx in range(args.eval.n_envs):
                        total_returns.append(total_return[idx])
                        total_costs.append(total_cost[idx])
                    break

    return total_returns,total_costs,np.sum(total_costs)/total_steps


def evaluate(actor, env, num_episodes=10, vis=True):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = []
    total_returns = []
    total_costs = []

    while len(total_returns) < num_episodes:
        state,_ = env.reset()
        done = False
        total_return = 0
        total_cost = 0
        total_timestep = 0
        with eval_mode(actor):
            while not done:
                action = actor.choose_action(state, sample=False)
                next_state, reward,cost, done,trunc, info = env.step(action)
                state = next_state
                total_return += reward
                total_cost += cost
                total_timestep += 1
                if (done or trunc):
                    total_returns.append(total_return)
                    total_costs.append(total_cost)
                    total_timesteps.append(total_timestep)
                    break

    return total_returns,total_costs, total_timesteps


def weighted_softmax(x, weights):
    x = x - torch.max(x, dim=0)[0]
    return weights * torch.exp(x) / torch.sum(
        weights * torch.exp(x), dim=0, keepdim=True)


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def mlp_withdropout(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, dropout=0.5):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def get_concat_samples(mix_batch, bad_batch, args):
    mix_batch_state, mix_batch_next_state, mix_batch_action, mix_batch_reward, mix_batch_cost, mix_batch_done, mix_batch_is_constrained = mix_batch
    bad_batch_state, bad_batch_next_state, bad_batch_action, bad_batch_reward, bad_batch_cost, bad_batch_done, bad_batch_is_constrained = bad_batch

    mix_batch_reward = torch.ones_like(mix_batch_reward)
    bad_batch_reward = torch.zeros_like(bad_batch_reward)

    batch_state         = torch.cat([mix_batch_state, bad_batch_state], dim=0)
    batch_next_state    = torch.cat([mix_batch_next_state, bad_batch_next_state], dim=0)
    batch_action        = torch.cat([mix_batch_action, bad_batch_action], dim=0)
    batch_reward        = torch.cat([mix_batch_reward, bad_batch_reward], dim=0)
    batch_cost          = torch.cat([mix_batch_cost, bad_batch_cost], dim=0)
    batch_done          = torch.cat([mix_batch_done, bad_batch_done], dim=0)
    batch_is_constrained= torch.cat([mix_batch_is_constrained, bad_batch_is_constrained], dim=0)
    is_bad              = torch.cat([torch.zeros_like(mix_batch_reward, dtype=torch.bool),
                           torch.ones_like(bad_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward,batch_cost, batch_done, batch_is_constrained, is_bad


def save_state(tensor, path, num_states=5):
    """Show stack framed of images consisting the state"""

    tensor = tensor[:num_states]
    B, C, H, W = tensor.shape
    images = tensor.reshape(-1, 1, H, W).cpu()
    save_image(images, path, nrow=num_states)
    # make_grid(images)


def average_dicts(dict1, dict2):
    return {key: round(1/2 * (dict1.get(key, 0) + dict2.get(key, 0)),3)
                     for key in set(dict1) | set(dict2)}