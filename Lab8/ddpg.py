'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))
    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super(ActorNet, self).__init__()
        ## TODO ##
        self.l1 = nn.Linear(state_dim, 400)
        self.l1.weight.data.normal_(0,0.1) # initialization
        self.l2 = nn.Linear(400, 300)
        self.l2.weight.data.normal_(0,0.1) # initialization
        self.l3 = nn.Linear(300, action_dim)
        self.l3.weight.data.normal_(0,0.1) # initialization
        # self.max_action = max_action
    def forward(self, x):
        ## TODO ##
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))# * self.max_action
        return a


class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.fcs1 = nn.Linear(state_dim,h1)
        self.fcs1.weight.data.normal_(0,0.1) # initialization
        self.fcs2 = nn.Linear(h1,h2)
        self.fcs2.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(action_dim,h2)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(h2,2)
        self.out.weight.data.normal_(0, 0.1)  # initialization
    def forward(self,s,a):
        x = self.fcs1(s)
        x = self.fcs2(x)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value
        
        
    #     h1, h2 = hidden_dim
    #     self.critic_head = nn.Sequential(
    #         nn.Linear(state_dim + action_dim, h1),
    #         nn.ReLU(),
    #     )
    #     self.critic = nn.Sequential(
    #         nn.Linear(h1, h2),
    #         nn.ReLU(),
    #         # nn.Linear(h2, action_dim),
    #         nn.Linear(h2, 1),
    #     )

    # def forward(self, x, action):
    #     x = self.critic_head(torch.cat([x, action], dim=1))
    #     return self.critic(x)


class DDPG:
    def __init__(self, args, max_action):
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net = CriticNet().to(args.device)
        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net = CriticNet().to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())
        ## TODO ##
        self._actor_opt = optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt = optim.Adam(self._critic_net.parameters(), lr=args.lrc)
        
        # action noise
        self.Gau_var = args.Gau_var
        self.Gau_decay = args.Gau_decay
        # self._action_noise = GaussianNoise(dim=2, mu=None, std=self.Gau_var)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.max_action = max_action #float(env.action_space.high[0])

    def select_action(self, state, noise_inp=True):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        state = torch.FloatTensor(state).to(self.device)
        next_action = self._actor_net(state).detach()
        if noise_inp == True:
            self._action_noise = GaussianNoise(dim=2, mu=None, std=self.Gau_var)
            noise = self._action_noise.sample()
            noise = torch.FloatTensor(noise).to(self.device)
            noise_action = next_action + noise
            noise_a_next = noise_action.clamp(-self.max_action, self.max_action)
            self.Gau_var *= self.Gau_decay
            return noise_a_next.cpu().data.numpy()
        else:
            return next_action.cpu().data.numpy()

    def append(self, state, action, reward, next_state, done):
        # self._memory.append(state, action, [reward / 100], next_state, [int(done)])
        self._memory.append(state, action, [reward], next_state, [int(done)])

    def update(self):
        # update the behavior networks
        self._update_behavior_network(self.gamma)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net, self.tau)
        self._update_target_network(self._target_critic_net, self._critic_net, self.tau)

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)

        ##### update critic #####
        # critic loss
        ## TODO ##
        a_next = target_actor_net(next_state).detach()
        q_target = reward + (1-done) * self.gamma * target_critic_net(next_state, a_next).detach()
        q_next = critic_net(state, action)
        loss_fn = nn.MSELoss()
        critic_loss = loss_fn(q_next, q_target)
        
        ### optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ##### update actor #####
        # actor loss
        ## TODO ##
        a_act = actor_net(state)
        q_act = critic_net(state, a_act)
        actor_loss = -torch.mean(q_act)
        
        ### optimize actor
        actor_net.zero_grad()
        critic_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            ## TODO ##
            target.data.copy_(tau * behavior.data + (1-tau) * target.data)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({'actor': self._actor_net.state_dict(),
                        'critic': self._critic_net.state_dict(),
                        'target_actor': self._target_actor_net.state_dict(),
                        'target_critic': self._target_critic_net.state_dict(),
                        'actor_opt': self._actor_opt.state_dict(),
                        'critic_opt': self._critic_opt.state_dict(),
                       }, model_path)
        else:
            torch.save({'actor': self._actor_net.state_dict(),
                        'critic': self._critic_net.state_dict(),
                        }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net.load_state_dict(model['critic'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt.load_state_dict(model['critic_opt'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
        # for t in range(300): #itertools.count(start=1): 
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update()

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward))
                if ewma_reward > 270:
                    model_fn = 'save_ddpg/DDPG_eps' +str(episode) +'reward' +str(np.round(ewma_reward,2)) +'.pth'
                    agent.save(model_fn)
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        total_steps = 0
        ## TODO ##
        for t in itertools.count(start=1): 
            # select action
            action = agent.select_action(state, noise_inp=False)
            # execute action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                break
        rewards.append(total_reward)
    print('Average Reward', np.mean(rewards))
    print(rewards)
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ddpg.pth')
    parser.add_argument('--logdir', default='log/ddpg')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=2000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.001, type=float)
    parser.add_argument('--Gau_var', default=3, type=float)
    parser.add_argument('--Gau_decay', default=0.99995, type=float)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')
    max_action = float(env.action_space.high[0])
    agent = DDPG(args, max_action)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
