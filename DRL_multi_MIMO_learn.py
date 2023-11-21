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
from DRL_env_MIMO import DRLenvMIMO
import scipy.special as sp

class DRLmultiMIMO(object):
    def __init__(self, state_size, action_size):


        self.antenna = 10
        self.users = 4
        self.user_selection_num = 2

        self.transmitters = 19

        #self.EPSILON = 0.1

        #self.EPSILON_DECAY = 1-math.pow(10,-4)
        #self.EPSILON_DECAY = 1
        #self.EPSILON_MIN = 0.01

        self.learning_rate = 5 * math.pow(10, -3)
        #self.learning_rate_decay = 1-math.pow(10,-4)

        self.gamma = 0.95

        self.pmax = 6.30957 #38dbm

        self.state_size = state_size
        self.action_size = action_size
        self.power_cand = 5
        self.power_set = np.linspace(0, self.pmax, self.power_cand)
        self.user_set = np.arange(0, self.users, 1)
        self.action_set_temp = np.arange(0, self.users * self.power_cand, 1)
        self.action_set = list(it.combinations(self.action_set_temp, self.user_selection_num))

        self.env = DRLenvMIMO()
        self.A = self.env.tx_positions_gen()
        self.B = self.env.rx_positions_gen(self.A)

        #self.H = np.ones((self.transmitters, self.antenna, self.users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        
        

        #self.action = np.zeros(action_size)



        self.noise = math.pow(10,-11.4)

        self.model = self.build_network

        self.replay_buffer = deque(maxlen=5000)
        self.update_rate = 100
        self.main_network = self.build_network()
        weight = self.main_network.get_weights()

        for i in range(1, self.transmitters + 1):
            setattr(self, f'target_network{i}', self.build_network())
            getattr(self, f'target_network{i}').set_weights(weight)





    def build_network(self):
        model = Sequential()
        model.add(Dense(200, activation="tanh", input_shape=(1,)))
        model.add(Dense(100, activation="tanh"))
        model.add(Dense(50, activation="tanh"))
        model.add(Dense(self.action_size, activation="tanh"))
        model.compile(loss="mse", optimizer='adam')

        return model

    def store_transistion(self, state, action, reward, next_state, done, agent):
        self.replay_buffer.append((state, action, reward, next_state, done, agent))

    def epsilon_greedy(self, agent, state, epsilon):
        if np.random.random() <= epsilon:
            action_temp = np.random.choice(len(self.action_set))
            action = self.action_set[int(action_temp)]

        else:
            Q_values = getattr(self, f'target_network{agent + 1}').predict(state)
            action_temp = np.argmax(Q_values[0])
            action = self.action_set[int(action_temp)]

        return action



    def full_csi(self, A, B, previous_full):
        H = np.zeros((self.antenna, self.users))

        for i in range(self.antenna):
            for j in range(self.users):
                temp = self.env.Jakes_channel(previous_full[:,j])
                temp_gain = self.env.channel_gain(A[i],B[i][j],temp)
                H[i, j] = temp_gain

        return H

    def scheduled_csi(self, selected_users,H):
        scheduled_H = np.zeros((self.antenna, self.user_selection_num))

        for j in range(self.user_selection_num):
            temp = selected_users[j]
            scheduled_H[:,j] = H[:, int(temp)]

        return scheduled_H


    def digital_precoder(self, Heq): # reduced Heq 입력 시 (즉, Heq = [KxK] )

        if np.linalg.det(np.matrix(Heq) @ np.conj(np.matrix(Heq)).T) == 0:
            before_inv = np.matrix(Heq) @ np.conj(np.matrix(Heq)).T
            before_inv[0, 0] += 0.0001

            F_BB = np.conj(np.matrix(Heq)).T @ np.linalg.inv(before_inv)
            F_BB = F_BB / np.linalg.norm(F_BB, 2)

        else:
            F_BB = np.conj(np.matrix(Heq)).T @ np.linalg.inv(np.matrix(Heq) @ np.conj(np.matrix(Heq)).T)
            F_BB = F_BB / np.linalg.norm(F_BB, 2)

        return F_BB


    def step(self, state, actions, TTI, max_TTI, agent, H):

        if TTI >= max_TTI:
            done = True
        else:
            done = False

        action_of_agent = actions[agent]
        powers_of_agent = np.zeros((self.user_selection_num))
        user_index_of_agent = np.zeros((self.user_selection_num))

        for i in range(self.user_selection_num):
            user_index_of_agent[i] = action_of_agent[i] % self.users
            powers_of_agent[i] = self.power_set[int(action_of_agent[i] // self.users)]

        selected_H = self.scheduled_csi(user_index_of_agent, H[agent, agent, :, :])
        direct_signal = np.zeros((self.user_selection_num))
        for i in range(self.user_selection_num):
            gain_temp = self.env.channel_gain(self.A[agent], self.B[agent][int(user_index_of_agent[i])], selected_H[:,i])
            F_bb = self.digital_precoder(selected_H[:,i])
            test = gain_temp @ F_bb * powers_of_agent[i]
            direct_signal[i] =test



        inter = np.zeros((self.users))
        for i in range(self.users):
            inter_temp_temp = 0
            for j in range(self.transmitters):
                if j == agent:
                    inter_temp_temp += 0
                else:
                    action_of_interferer = actions[j]
                    user_index_of_interferer = np.zeros((self.user_selection_num))
                    power_of_interferer = np.zeros((self.user_selection_num))
                    for k in range(self.user_selection_num):
                        user_index_of_interferer[k] = action_of_interferer[k] % self.users
                        power_of_interferer[k] = self.power_set[int(action_of_interferer[k] // self.users)]
                    selected_H_interferer = self.scheduled_csi(user_index_of_interferer, H[j, j, :, :])
                    Fbb_interferer = np.zeros((self.user_selection_num, self.antenna))
                    for k in range(self.user_selection_num):
                        Fbb_interferer[k, :] = np.array(self.digital_precoder(selected_H_interferer[:,k])).flatten()

                    for k in range(self.user_selection_num):
                        gain_temp_interferer = self.env.channel_gain(self.A[j], self.B[agent][k], H[j, agent, :, i])
                        inter_of_interferer = gain_temp_interferer @ Fbb_interferer[k,:] * power_of_interferer[k]
                        inter_temp_temp += inter_of_interferer

            inter[i] = inter_temp_temp


        next_state = inter


        sum_rate = 0
        reward = 0

        for i in range(self.user_selection_num):
            SINR_temp = (np.abs(direct_signal[i]))/(np.abs(state[int(user_index_of_agent[i])]) + self.noise)
            reward += math.log(1+SINR_temp)

        info = {}

        agent = agent

        return  next_state, reward, done, info, agent


    def train(self, batch_size):

        # sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        # compute the Q value using the target network
        for state, action, reward, next_state, done, agent in minibatch:
            if not done:
                target_Q = reward + self.gamma * np.amax(getattr(self, f'target_network{agent+1}').predict(next_state))

            else:
                target_Q = reward

            # compute the Q value using the main network
            Q_values = self.main_network.predict(state)

            for i in range(len(self.action_set)):
                if action == self.action_set[i]:
                    action_node_number = i


            Q_values[0][action_node_number] = target_Q

            # train the main network
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)




    def update_target_network(self):
        weight = self.main_network.get_weights()
        for i in range(1, self.transmitters + 1):
            getattr(self, f'target_network{i}').set_weights(weight)
        return 0



