import math
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from collections import deque
import time
from DRL_env import DRLenv

'''
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys
'''
'''
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

if use_cuda:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


if use_cuda:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
'''
class DRLagent(object):
    def __init__(self, state_size, action_size):
        self.TTIs = 2000
        self.simul_rounds = 100

        self.EPSILON = 0.1

        #self.EPSILON_DECAY = 0.99
        self.EPSILON_DECAY = 1
        self.EPSILON_MIN = 0.01

        self.learning_rate = 5 * math.pow(10, -3)
        #self.learning_rate_decay = 1-math.pow(10,-4)

        self.gamma = 0.95

        self.pmax = 6.30957 #38dbm

        self.state_size = state_size
        self.action_size = action_size
        self.action_cand = 10
        self.action_set = np.linspace(0, self.pmax, self.action_cand)

        self.env = DRLenv()
        self.A = self.env.tx_positions_gen()
        self.B = self.env.rx_positions_gen(self.A)

        self.H = np.ones((state_size,state_size)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)



        #self.action = np.zeros(action_size)



        self.noise = math.pow(10,-14.4)

        self.model = self.build_network

        self.replay_buffer = deque(maxlen=5000)
        self.update_rate = 100
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())





    def build_network(self):
        model = Sequential()
        model.add(Dense(6000, activation="tanh", input_shape=(1,)))
        model.add(Dense(4000, activation="tanh"))
        model.add(Dense(2000, activation="tanh"))
        model.add(Dense(self.action_size, activation="tanh"))
        model.compile(loss="mse", optimizer='adam')

        return model

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        if np.random.random() <= self.EPSILON:
            actions = np.zeros(3)
            for i in range(3):
                action_temp = np.random.randint(self.action_cand)
                actions[i] = self.action_set[action_temp]

        else:
            Q_values = self.main_network.predict(state)
            actions = np.zeros(3)
            action_temp2 = np.argmax(Q_values[0])
            test = action_temp2
            for j in range(3):
                actions[2-j] = self.action_set[test % self.action_cand]
                test = test // self.action_cand


        return actions

    def full_csi(self, A, B, previous):
        H = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                temp = self.env.Jakes_channel(previous[i,j])
                temp_gain = self.env.channel_gain(A[i],B[j],temp)
                H[i, j] = temp_gain

        return H
    '''
    def SE(self,action, agent, TTI):
        H = self.full_csi(TTI)
        inter_temp = 0
        for i in range(3):
            inter_temp += H[i, agent] * action[i]

        inter = inter_temp - (H[agent, agent]* action[agent])

        SINR = (H[agent, agent] * action[agent])/(inter + self.noise)

        rate = math.log(1+SINR)

        return rate
    '''


    def step(self, state, action, TTI, max_TTI):

        if TTI > max_TTI:
            done = True
        else:
            done = False

        self.H = self.full_csi(self.A, self.B, self.H)

        inters = np.zeros(3)

        for i in range(3):

            inter_temp = 0
            for j in range(3):
                inter_temp += self.H[j, i] * action[j]

            inter = inter_temp - (self.H[i, i] * action[i])

            inters[i] =inter

        sum_rate = 0
        reward = 0

        for k in range(3):
            last_inter_temp = np.sum(state) - state[k]
            SINR_temp = (self.H[k, k] * action[k])/(last_inter_temp + self.noise)
            reward_temp = math.log(1+SINR_temp)
            reward += reward_temp

        next_state = inters

        info = {}

        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

        return  next_state, reward, done, info



    def train(self, batch_size):

        # sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)


        # compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
            else:
                target_Q = reward

            # compute the Q value using the main network
            Q_values = self.main_network.predict(state)

            action_number = 0
            action_change = np.zeros((self.state_size))
            for i in range(self.state_size):
                stepping = self.pmax / (self.action_cand-1)
                action_change[i] = math.floor(action[i]/stepping)


            for i in range(self.state_size):
                action_number += action_change[i] * (self.action_cand ** (2-i))

            action_number = int(action_number)

            Q_values[0][action_number] = target_Q

            # train the main network
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)

        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY



    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())



    '''
    def choose_action(self, epsilon):

        actions = np.zeros(3)

        if np.random.random() <= epsilon:
            for i in range(3):
                actions[i] = np.random.choice(3)

            return actions


        else:
            for i in range(3):
                actions[i] = np.argmax(q[i])

            return actions


    '''




