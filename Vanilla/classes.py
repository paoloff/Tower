import numpy as np
import torch
import pandas as pd
import yfinance as yf
import gym
import copy
import random
from collections import deque


class Data:
    def __init__(self, assets):
        self.assets = assets
        self.tickers = dict(zip(self.assets, [yf.Ticker(name) for name in self.assets]))
        

    def time_series(self,  interval = "1d", period = "1mo"): 
        open = []
        close = []
        for i in range(len(self.assets)):
            open.append(np.array(self.tickers[self.assets[i]].history(period = period, interval = interval)["Open"]))
            close.append(np.array(self.tickers[self.assets[i]].history(period = period, interval = interval)["Close"]))
        
        return open, close

class Portfolio():
    def __init__(self, cash, assets, allocations, interest_rate = 10**(-8)):
        self.cash = cash
        self.assets = dict(zip(assets, allocations))
        self.allocations = allocations
        self.interest_rate = interest_rate
        self.nav = cash

    def update(self, open, close, signal):
        open_nav = np.sum(self.allocations*open) + self.cash
        self.cash = (self.cash - np.sum(open*signal))*(1+self.interest_rate)
        close_nav = np.sum((self.allocations + signal)*close) + self.cash
        self.allocations += signal
        return close_nav - open_nav, close, self.allocations, self.cash, close_nav

    def exit(self, prices):
        self.update(prices, -self.allocations)


class MarketSimulator(gym.Env):
    def __init__(self, cash, assets, allocations, interest_rate, interval, period, steps_per_episode):
        self.time_step = 0
        self.initial_cash = cash
        self.n_assets = len(assets)
        self.portfolio = Portfolio(cash, assets, allocations, interest_rate)
        self.data = Data(assets)
        self.open  = self.data.time_series(interval, period)[0]
        self.close_ = self.data.time_series(interval, period)[1]
        self.max_number_of_steps = steps_per_episode
        self.data_length = len(self.open[0])
        self.initial_allocations = np.zeros(self.n_assets)
        self.interest_rate = interest_rate
        self.assets = assets

    def step(self, action):
        open_prices = np.array([self.open[i][self.time_step] for i in range(self.n_assets)])
        close_prices = np.array([self.close_[i][self.time_step] for i in range(self.n_assets)])
        self.time_step += 1
        return self.portfolio.update(open_prices, close_prices, action)

    def reset(self, cash, assets, allocations):
        self.portfolio = Portfolio(self.initial_cash, self.assets, self.initial_allocations, self.interest_rate)
        #self.portfolio = Portfolio(cash, assets, allocations)
        self.time_step = random.randint(0, self.data_length - self.max_number_of_steps - 1)
        #self.time_step = 500

    

class QNN(torch.nn.Module):
    def __init__(self, n_assets, n1=3, n2=30, lr=0.05, momentum=0.9):
        super(QNN, self).__init__()
        self.linear1 = torch.nn.Linear(2*n_assets + 1, n1)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n1, n2)
        self.linear3 = torch.nn.Linear(n2, 3**n_assets)
        self.optimizer = torch.optim.SGD(self.parameters(), lr, momentum)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)

        return x
    
    def train_on_batch(self, x, y):
        pred = self.forward(x)
        loss_function = torch.nn.MSELoss()
        loss = loss_function(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss


class RfLearner():
    def __init__(self, assets, epsilon=0.1, batch_size=20, gamma=0.9, experience_buffer_size=50, tau=15):
        self.epsilon = epsilon
        self.n_assets = len(assets)
        self.online_net = QNN(self.n_assets)
        self.target_net = QNN(self.n_assets)
        self.batch_size = batch_size
        self.gamma = gamma
        self.losses = []
        self.tau = tau
        self.experience_buffer_size = experience_buffer_size
        self.experience = deque([], self.experience_buffer_size)
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_length = 0

    def epsilon_greedy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return torch.tensor(random.choice(range(3**self.n_assets)))
        q = self.online_net.forward(state)
        return torch.argmax(q, axis = 0)
    
    def action_number(self, action):
        # action = torch.reshape(action, (self.batch_size, 2))
        digits = np.mod(action, 3)
        action_numbers = torch.zeros(self.batch_size)
        for j in range(self.batch_size): action_numbers[j]= np.sum([digits[j][i]*3**(self.n_assets-1-i) for i in range(self.n_assets)])
        return action_numbers        
    
    def update_target(self):
        self.target_net = copy.deepcopy(self.online_net)
    
    def memorize(self, state, action, reward, next_state, done=0):
        if not done:
            self.episode_reward += reward
            self.episode_length += 1
        
        else:
            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((state, action, reward, next_state))
    
    def experience_replay(self):
        if self.batch_size > len(self.experience): return
        minibatch = random.sample(self.experience, self.batch_size)
        states = torch.stack([minibatch[i][0] for i in range(self.batch_size)],0)
        action_numbers = self.action_number([minibatch[i][1] for i in range(self.batch_size)])
        rewards = torch.tensor([minibatch[i][2] for i in range(self.batch_size)])
        next_states = torch.stack([minibatch[i][3] for i in range(self.batch_size)],0)
        qs_next = self.online_net.forward(next_states)
        best_actions = torch.argmax(qs_next, axis=1)
        qs_next_target = self.target_net.forward(next_states)
        target_qs = torch.tensor([qs_next_target[i][int(best_actions[i])] for i in range(self.batch_size)])
        targets = rewards + self.gamma*target_qs
        qs_current = self.online_net.forward(states)
        for i in range(self.batch_size): qs_current[i][int(action_numbers[i])] = targets[i]
        loss = self.online_net.train_on_batch(states, qs_current)
        self.losses.append(loss)

        if self.total_steps % self.tau==0:
            self.update_target()
    
      
    
    
    
