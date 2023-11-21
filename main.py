import tensorflow as tf
from DRL_env import DRLenv
from DRL_learn import DRLagent
from DRL_multi_learn import DRLmultiagent
from DRL_multi_MIMO_learn import DRLmultiMIMO
from DRL_env_MIMO import DRLenvMIMO
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import math
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from scipy.optimize import minimize
import time
import sys
import itertools as it

import random
import scipy.special as sp



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

'''
def main():
    num_TTIs = 500
    num_simul_rounds = 1

    batch_size = 8
    env = DRLenv()
    dqn = DRLagent(3, 1000)


    done = False
    TTI = 0

    rewards = np.zeros((num_simul_rounds, num_TTIs))

    for i in range(num_simul_rounds):
        Return = 0
        state = np.zeros(3)
        action = np.zeros(3)

        for j in range(num_TTIs):



            if j % dqn.update_rate == 0:
                dqn.update_target_network()

            starter = time.time()
            action = dqn.epsilon_greedy(state)
            end = time.time()
            print('time =', end-starter)
            next_state, reward, done, _ = dqn.step(state, action, j, num_TTIs)

            dqn.store_transistion(state, action, reward, next_state, done)

            state = next_state

            Return += reward
            '''
            if j == 0:
                cumul_reward[i, j] = Return
            else:
                cumul_reward[i, j] = Return/j
            '''
            rewards[i, j] = reward
            print('next_state', next_state, 'action', action)

            print('Iteration:', j, ',' 'Reward', reward)

            if done:
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn.replay_buffer) > batch_size:
                dqn.train(batch_size)

    reward_avg = rewards.sum(axis=0) / num_simul_rounds

    #np.save('./save_weights/centralized_DRL.npy', rewards)
    #np.save('./save_weights/centralized_DRL_test.npy', rewards)




def main_multi_MIMO():
    num_simul_rounds = 1
    num_TTIs = 2000

    batch_size = 8
    env = DRLenvMIMO()
    dqn_multi = DRLmultiMIMO(19, 190)

    done = False
    TTI = 0

    rewards = np.zeros((num_simul_rounds, num_TTIs))

    f_d = 10
    T = 0.2
    rho = sp.jv(0, 2*math.pi*f_d*T)
    transmitters = 19
    cell = 19
    antenna = 10

    users = 4
    power_cand = 10
    pmax = 6.30957 #38dbm
    user_selection_num = 2
    power_set = np.linspace(0, pmax, power_cand)
    user_set = np.arange(0, users, 1)
    action_set_temp = np.arange(0, users * power_cand, 1)
    action_set = list(it.combinations(action_set_temp, user_selection_num))

    for i in range(num_simul_rounds):
        Return = 0

        states_of_agents = np.zeros((transmitters, users))
        actions_of_agents = []
        for j in range(transmitters):
            actions_of_agents.append((0,0))
        '''
        actions_of_agents_opt = []
        for j in range(transmitters):
            actions_of_agents_opt.append(0)
        actions_of_agents_opt_delay = []
        for j in range(transmitters):
            actions_of_agents_opt_delay.append(0)
        '''
        H = np.ones((transmitters, cell, antenna, users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        #prev_H = np.ones((transmitters, cell, antenna, users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        epsilon  = 0.1

        #epsilon_decay = 1-math.pow(10,-4)
        epsilon_decay = 1
        epsilon_min = 0.01
        for j in range(num_TTIs):

            if j % dqn_multi.update_rate == 0:
                dqn_multi.update_target_network()

            for k in range(transmitters):
                actions_of_agents[k] = dqn_multi.epsilon_greedy(k, states_of_agents[k,:], epsilon)

            for x in range(transmitters):
                for y in range(cell):
                    for z in range(antenna):
                        for w in range(users):
                            innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                            htemp = rho * H[x,y,z,w] + (math.sqrt(1-math.pow(rho, 2)) * innov)
                            H[x,y,z,w] = htemp

            reward_temp=np.zeros((transmitters))
            for k in range(transmitters):
                next_state, reward, done, info, agent = dqn_multi.step(states_of_agents[k,:], actions_of_agents, j, num_TTIs, k, H)
                dqn_multi.store_transistion(states_of_agents[k,:], actions_of_agents[k], reward, next_state, done, agent)
                states_of_agents[k,:] = next_state
                reward_temp[k] = reward

            final_reward = np.sum(reward_temp)

            for x in range(transmitters):
                for y in range(cell):
                    for z in range(antenna):
                        for w in range(users):
                            innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                            htemp = rho * H[x,y,z,w] + (math.sqrt(1-math.pow(rho, 2)) * innov)
                            H[x,y,z,w] = htemp

            Return += final_reward
            '''
            if j == 0:
                cumul_reward[i, j] = Return
            else:
                cumul_reward[i, j] = Return/j
            '''
            rewards[i, j] = final_reward
            #print('next_state', next_state, 'action', action)

            print('Iteration:', j, ',' 'Reward', final_reward)

            if done:
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn_multi.replay_buffer) > batch_size:
                dqn_multi.train(batch_size)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

    #np.save('./save_weights/multi_agent_DRL_MIMO.npy', rewards)
    #np.save('./save_weights/multi_agent_DRL_test.npy', rewards)

'''
def opt_MIMO():
    num_simul_rounds = 1
    num_TTIs = 1000

    batch_size = 8
    env = DRLenvMIMO()
    dqn_multi = DRLmultiMIMO(19, 190)

    done = False
    TTI = 0

    rewards = np.zeros((num_simul_rounds, num_TTIs))

    f_d = 10
    T = 0.2
    rho = sp.jv(0, 2*math.pi*f_d*T)

    self.pmax = 6.30957 #38dbm


    power_cand = 5
    power_set = np.linspace(0, self.pmax, self.power_cand)
    self.user_set = np.arange(0, self.users, 1)
        self.action_set_temp = np.arange(0, self.users * self.power_cand, 1)
        self.action_set = list(it.combinations(self.action_set_temp, self.user_selection_num))

    for i in range(num_simul_rounds):
        Return = 0
        transmitters = 19
        cell = 19
        antenna = 10
        users = 4



        states_of_agents = np.zeros((transmitters, users))
        actions_of_agents = []
        for j in range(transmitters):
            actions_of_agents.append((0,0))



        H = np.ones((transmitters, cell, antenna, users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        prev_H = np.ones((transmitters, cell, antenna, users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)

        for j in range(num_TTIs):

            for x in range(transmitters):
                for y in range(cell):
                    for z in range(antenna):
                        for w in range(users):
                            innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                            htemp = rho * H[x,y,z,w] + (math.sqrt(1-math.pow(rho, 2)) * innov)
                            H[x,y,z,w] = htemp

            optimal[i, j] = 0
            for k in range(transmitters):
                action_of_agent = actions[k]
                powers_of_agent = np.zeros((user_selection_num))
                user_index_of_agent = np.zeros((user_selection_num))

        for i in range(self.user_selection_num):
            user_index_of_agent[i] = action_of_agent[i] % self.users
            powers_of_agent[i] = self.power_set[int(action_of_agent[i] // self.users)]

        selected_H = self.scheduled_csi(user_index_of_agent, H[agent, agent, :, :])
        direct_signal = np.zeros((self.user_selection_num))
        for i in range(self.user_selection_num):
            gain_temp = self.env.channel_gain(self.A[agent], self.B[agent][int(user_index_of_agent[i])], selected_H[:,i])
            F_bb = self.digital_precoder(selected_H[:,i])
            direct_signal[i] = gain_temp @ F_bb * powers_of_agent[i]



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

            best1 = 0
            best2 = 0
            best3 = 0



            final_reward = np.sum(reward_temp)

            Return += final_reward

            rewards[i, j] = final_reward
            #print('next_state', next_state, 'action', action)

            print('Iteration:', j, ',' 'Reward', final_reward)

            if done:
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn_multi.replay_buffer) > batch_size:
                dqn_multi.train(batch_size)

    reward_avg = rewards.sum(axis=0) / num_simul_rounds

    np.save('./save_weights/multi_agent_DRL_MIMO.npy', rewards)
    #np.save('./save_weights/multi_agent_DRL_test.npy', rewards)


'''

def main_multi():
    num_simul_rounds = 1
    num_TTIs = 700

    batch_size = 8
    env = DRLenv()


    done = False


    f_d = 10
    T = 0.02
    rho = sp.jv(0, 2*math.pi*f_d*T)
    transmitters = 3
    users = 3
    pmax = 9 #38dbm
    action_cand = 10
    action_set = np.linspace(0, pmax, action_cand)
    noise = math.pow(10,-11.4) #-11.4)

    dqn_multi = DRLmultiagent(transmitters * users, 10, action_cand, pmax, noise)

    rewards = np.zeros((num_simul_rounds, num_TTIs))
    optimal = np.zeros((num_simul_rounds, num_TTIs))
    optimal_no_delay = np.zeros((num_simul_rounds, num_TTIs))
    full_pwr = np.zeros((num_simul_rounds, num_TTIs))
    random_pwr = np.zeros((num_simul_rounds, num_TTIs))

    for i in range(num_simul_rounds):
        Return = 0
        states_of_agents = np.zeros((transmitters, transmitters))#.flatten()
        #states_of_agents = tf.convert_to_tensor(states_of_agents.reshape(1, -1), dtype=tf.float32)

        actions_of_agents = np.zeros((transmitters))

        H = np.ones((transmitters, transmitters)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        #prev_H = np.ones((transmitters, cell, antenna, users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        epsilon  = 1

        epsilon_decay = 1-math.pow(10,-2)
        #epsilon_decay = 0.99
        epsilon_min = 0.01

        best1 = 0
        best2 = 0
        best3 = 0

        for j in range(num_TTIs):
            if j % dqn_multi.update_rate == 0:
                dqn_multi.update_target_network()

            for k in range(transmitters):
                actions_of_agents[k] = dqn_multi.epsilon_greedy(k, states_of_agents[k, :], epsilon)


            for x in range(transmitters):
                for y in range(users):
                    innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                    htemp = rho * H[x,y] + (math.sqrt(1-math.pow(rho, 2)) * innov)
                    H[x,y] = htemp

            channel_gain = np.zeros((transmitters, users))
            for x in range(transmitters):
                for y in range(users):
                    channel_gain[x, y] = env.channel_gain(dqn_multi.A[x], dqn_multi.B[y], H[x, y])

            next_state, SE, done, reward = dqn_multi.step(states_of_agents, actions_of_agents, j, num_TTIs, channel_gain)
            for k in range(transmitters):
                dqn_multi.store_transistion(states_of_agents[k, :], actions_of_agents[k], reward[k], next_state[k, :], done, k)
                        #### reward SE 주의!! 공사 중 !! ####
            final_reward = np.sum(SE)
            states_of_agents = next_state
            Return += final_reward
            rewards[i, j] = final_reward

            delayed_SE_1 = math.log2(1 + (channel_gain[0, 0] * best1) / (channel_gain[1, 0] * best2 + channel_gain[2, 0] * best3 + noise))

            delayed_SE_2 = math.log2(1 + (channel_gain[1, 1] * best2) / (channel_gain[0, 1] * best1 + channel_gain[2, 1] * best3 + noise))

            delayed_SE_3 = math.log2(1 + (channel_gain[2, 2] * best3) / (channel_gain[0, 2] * best1 + channel_gain[1, 2] * best2 + noise))

            optimal[i, j] = delayed_SE_1 + delayed_SE_2 + delayed_SE_3

            for k in range(action_cand):
                for l in range(action_cand):
                    for m in range(action_cand):
                        x = action_set[m]
                        y = action_set[l]
                        z = action_set[k]

                        OPT_SE_1 = math.log2(1 + (channel_gain[0, 0] * x) / (channel_gain[1, 0] * y + channel_gain[2, 0] * z + noise))

                        OPT_SE_2 = math.log2(1 + (channel_gain[1, 1] * y) / (channel_gain[0, 1] * x + channel_gain[2, 1] * z + noise))

                        OPT_SE_3 = math.log2(1 + (channel_gain[2, 2] * z) / (channel_gain[0, 2] * x + channel_gain[1, 2] * y + noise))

                        optimal_temp = OPT_SE_1 + OPT_SE_2 + OPT_SE_3

                        if optimal_temp > optimal_no_delay[i, j]:
                            optimal_no_delay[i, j] = optimal_temp
                            best1 = x
                            best2 = y
                            best3 = z


            full_SE_1 = math.log2(1 + (channel_gain[0, 0] * pmax) / (channel_gain[1, 0] * pmax + channel_gain[2, 0] * pmax + noise))

            full_SE_2 = math.log2(1 + (channel_gain[1, 1] * pmax) / (channel_gain[0, 1] * pmax + channel_gain[2, 1] * pmax + noise))

            full_SE_3 = math.log2(1 + (channel_gain[2, 2] * pmax) / (channel_gain[0, 2] * pmax + channel_gain[1, 2] * pmax + noise))

            full_pwr[i, j] = full_SE_1 + full_SE_2 + full_SE_3

            random_SE_1 = math.log2(
                1 + (channel_gain[0, 0] * action_set[random.randint(0, action_cand - 1)]) / (
                            channel_gain[1, 0] * action_set[random.randint(0, action_cand - 1)] + channel_gain[2, 0] *
                            action_set[random.randint(0, action_cand - 1)] + noise))

            random_SE_2 = math.log2(
                1 + (channel_gain[1, 1] * action_set[random.randint(0, action_cand - 1)]) / (
                            channel_gain[0, 1] * action_set[random.randint(0, action_cand - 1)] + channel_gain[2, 1] *
                            action_set[random.randint(0, action_cand - 1)] + noise))

            random_SE_3 = math.log2(
                1 + (channel_gain[2, 2] * action_set[random.randint(0, action_cand - 1)]) / (
                            channel_gain[0, 2] * action_set[random.randint(0, action_cand - 1)] + channel_gain[1, 2] *
                            action_set[random.randint(0, action_cand - 1)] + noise))

            random_pwr[i, j] = random_SE_1 + random_SE_2 + random_SE_3


            '''
            if j == 0:
                cumul_reward[i, j] = Return
            else:
                cumul_reward[i, j] = Return/j
            '''

            #print('next_state', next_state, 'action', action)

            print('Iteration:', j, ',' 'Reward', final_reward)
            print('Iteration:', j, ',' 'OPT Reward', optimal[i,j])
            print('Iteration:', j, ',' 'OPT (no delay) Reward', optimal_no_delay[i,j])
            print('Iteration:', j, ',' 'Full Power Reward', full_pwr[i, j])
            print('Iteration:', j, ',' 'Random Power Reward', random_pwr[i, j])
            if done: # 같은 TTI의 step func에서도 done은 세번 갱신된다.
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn_multi.replay_buffer) > batch_size:
                dqn_multi.train(batch_size)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            dqn_multi.learning_rate *= (1-math.pow(10, -4))


    reward_avg = rewards.sum(axis=0) / num_simul_rounds



    #np.save('./save_weights/FP.npy', optimal)
    np.save('./save_weights/full_power.npy', full_pwr)
    np.save('./save_weights/random_power.npy', random_pwr)
    np.save('./save_weights/multi_agent_DRL.npy', rewards)
    np.save('./save_weights/optimal_no_delay.npy', optimal_no_delay)
    np.save('./save_weights/optimal.npy', optimal)
    #np.save('./save_weights/multi_agent_DRL_test.npy', rewards)

    plt.plot(dqn_multi.loss)
    plt.show()

def fractional():
    num_TTIs = 2000
    num_simul_rounds = 1

    dqn = DRLagent(3, 1000)
    noise = math.pow(10,-11.4)
    pmax = 6.30957

    optimal = np.zeros((num_simul_rounds, num_TTIs))

    starter = time.time()
    for i in range(num_simul_rounds):
        for j in tqdm(range(num_TTIs)):
            H = np.zeros((3, 3))
            env = DRLenv(j+1)

            A = env.tx_positions_gen()
            B = env.rx_positions_gen(A)

            for l in range(3):
                for m in range(3):
                    temp = env.Jakes_channel(A[l], B[m])
                    H[l, m] = temp
            print(H)
            finish = False
            p = [pmax, pmax, pmax]
            trans = np.zeros(3)
            last_solution = 1
            sol = 0
            timer = 0
            while timer<1000:

                for n in range(3):
                    trans_temp = 0
                    for o in range(3):
                        trans_temp += H[o, n] * p[o]
                    trans_temp2 = trans_temp - (H[n, n]* p[n])

                    trans[n] = (H[n,n]*p[n])/(trans_temp2 + noise)

                print(trans)

                x = cp.Variable()
                y = cp.Variable()
                z = cp.Variable()
                rate1 = cp.log(1+2*cp.multiply(trans[0] , cp.sqrt(cp.multiply(H[0,0], x))) - cp.multiply(trans[0], trans[0]) * (cp.multiply(H[1,0], y)+cp.multiply(H[2,0], z)+noise))
                rate2 = cp.log(1+2*cp.multiply(trans[1] , cp.sqrt(cp.multiply(H[1,1], y))) - cp.multiply(trans[1], trans[1]) * (cp.multiply(H[0,1], x)+cp.multiply(H[2,1], z)+noise))
                rate3 = cp.log(1+2*cp.multiply(trans[2] , cp.sqrt(cp.multiply(H[2,2], z))) - cp.multiply(trans[2], trans[2]) * (cp.multiply(H[0,2], x)+cp.multiply(H[1,2], y)+noise))
                objective_func = cp.Maximize(rate1+rate2+rate3)
                #obj = cp.Maximize(cp.multiply(H[0,0] , x)/(cp.multiply(H[1,0] , y)  + H[2,0] + noise))
                con = [0 <= x, x<= pmax, 0 <= y, y<= pmax, 0 <= z, z<= pmax]
                #con = [0 <= x, x<= pmax, 0 <= y, y<= pmax]
                prob = cp.Problem(objective_func, con)
                sol = prob.solve()
                p[0] = x.value
                p[1] = y.value
                p[2] = z.value
                '''
                if np.absolute((sol-last_solution)/(last_solution+0.0000001)) <= 0.001:
                    finish = True

                last_solution = sol
                '''
                timer += 1
            optimal[i, j] = sol

    end = time.time()

    average_time = (end-starter)/(num_simul_rounds*num_TTIs)
    np.save('./save_weights/FP2.npy', optimal)
    print(optimal)

'''
def objective(x):
    return -math.log(1+(H[0,0] * x[0])/(H[1,0] * x[1] + H[2,0] * x[2] + noise)) - math.log(1+(H[1,1] * x[2])/(H[0,1] * x[0] + H[2,1] * x[2] + noise)) - math.log(1+ (H[2,2] * x[2])/(H[0,2] * x[0] + H[1,2] * x[1] + noise))



def fractional2():
    num_TTIs = 2000
    num_simul_rounds = 1

    dqn = DRLagent(3, 64)
    noise = math.pow(10,-11.4)
    pmax = 6.30957

    optimal = np.zeros((num_simul_rounds, num_TTIs))

    for i in range(num_simul_rounds):
        for j in range(num_TTIs):
            H = np.zeros((3, 3))
            env = DRLenv(j+1)

            A = env.tx_positions_gen()
            B = env.rx_positions_gen(A)

            for l in range(3):
                for m in range(3):
                    temp = env.Jakes_channel(A[l], B[m])
                    H[l, m] = temp

            starting_point = [0, 0, 0]
            x =

            result = minimize(objective, starting_point)

            solution = result['x']
            evaluation = -1 * objective(solution)

            optimal[i,j] = evaluation

    print(evaluation)
'''

def opt():
    num_TTIs = 1000
    num_simul_rounds = 1

    dqn = DRLagent(3, 64)
    noise = math.pow(10,-11.4)
    pmax = 6.30957

    action_cand = 10
    action_set = np.linspace(0, pmax, action_cand)

    optimal = np.zeros((num_simul_rounds, num_TTIs))
    start = time.time()

    env = DRLenv()
    A = env.tx_positions_gen()
    B = env.rx_positions_gen(A)
    temp = np.ones((3,3)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
    H= np.zeros((3,3))

    for i in tqdm(range(num_simul_rounds)):
        for j in range(num_TTIs):





            for l in range(3):
                for m in range(3):
                    temp[l,m] = env.Jakes_channel(temp[l,m])
                    H[l, m] = env.channel_gain(A[l],B[m],temp[l,m])
            print(H)

            optimal[i, j] = 0
            best1 = 0
            best2 = 0
            best3 = 0
            for k in range(action_cand):
                for l in range(action_cand):
                    for m in range(action_cand):
                        x = action_set[m]
                        y = action_set[l]
                        z = action_set[k]
                        optimal_temp = math.log(1+(H[0,0] * x)/(H[1,0] * y + H[2,0] * z + noise)) + math.log(1+(H[1,1] * y)/(H[0,1] * x + H[2,1] * z + noise)) + math.log(1+ (H[2,2] * z)/(H[0,2] * x + H[1,2] * y + noise))




                        if optimal_temp > optimal[i, j]:
                            optimal[i, j] = optimal_temp
                            best1 = x
                            best2 = y
                            best3 = z
            print(best2)
            print(best2)
            print(best3)
    end = time.time()
    average_time = (end-start)/(num_simul_rounds*num_TTIs)

    print('opt_time = ', average_time)
    print(optimal)



    #np.save('./save_weights/FP.npy', optimal)
    np.save('./save_weights/FP_test.npy', optimal)

def full_pwr():
    num_TTIs = 1000
    num_simul_rounds = 1

    dqn_multi = DRLmultiagent(9, 10)

    noise = math.pow(10,-11.4)
    pmax = 6.30957

    action_cand = 10


    full_pwr = np.zeros((num_simul_rounds, num_TTIs))


    env = DRLenv()
    A = env.tx_positions_gen()
    B = env.rx_positions_gen(A)
    temp = np.ones((3,3)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
    H= np.zeros((3,3))

    for i in tqdm(range(num_simul_rounds)):
        for j in range(num_TTIs):

            for l in range(3):
                for m in range(3):
                    temp[l,m] = env.Jakes_channel(temp[l,m])
                    H[l, m] = env.channel_gain(A[l],B[m],temp[l,m])

            SNR_1 = math.log2(1 + H[0, 0] * pmax / (H[1, 0] * pmax + H[2, 0] * pmax + noise))
            SNR_2 = math.log2(1 + H[1, 1] * pmax / (H[0, 1] * pmax + H[2, 1] * pmax + noise))
            SNR_3 = math.log2(1 + H[2, 2] * pmax / (H[0, 2] * pmax + H[1, 2] * pmax + noise))

            full_pwr[i, j] = SNR_1 + SNR_2 + SNR_3

    print(full_pwr)
    #np.save('./save_weights/FP.npy', optimal)
    np.save('./save_weights/full_power.npy', full_pwr)




def graph():

    centralized_DRL = np.load('./save_weights/centralized_DRL_test.npy')
    multi_agent_DRL = np.load('./save_weights/multi_agent_DRL.npy')
    multi_agent_DRL_MIMO = np.load('./save_weights/multi_agent_DRL_MIMO.npy')
    FP = np.load('./save_weights/FP_test.npy')
    optimal = np.load('./save_weights/optimal.npy')
    optimal_no_delay = np.load('./save_weights/optimal_no_delay.npy')
    full_pwr = np.load('./save_weights/full_power.npy')
    random_pwr = np.load('./save_weights/random_power.npy')

    num_TTIs = 500
    num_simul_rounds = 1
    start = 20

    reward_avg = centralized_DRL.sum(axis=0) / num_simul_rounds
    reward_avg_multi = multi_agent_DRL.sum(axis=0) / num_simul_rounds
    reward_avg_multi_MIMO = multi_agent_DRL_MIMO.sum(axis=0) / num_simul_rounds
    reward_avg_FP = FP.sum(axis=0) / num_simul_rounds
    reward_avg_optimal = optimal.sum(axis=0) / num_simul_rounds
    reward_avg_optimal_no_delay = optimal_no_delay.sum(axis=0) / num_simul_rounds
    reward_avg_full_pwr = full_pwr.sum(axis=0) / num_simul_rounds
    reward_avg_random_pwr = random_pwr.sum(axis=0) / num_simul_rounds

    for i in range(len(reward_avg)):
        if reward_avg[i] ==0 and i != 0:
            reward_avg[i] = reward_avg[i-1]

    for i in range(len(reward_avg_multi)):
        if reward_avg_multi[i] ==0 and i != 0:
            reward_avg_multi[i] = reward_avg_multi[i-1]

    cumulative_rewards = [np.mean(reward_avg[:i + 1]) for i in range(start, len(reward_avg))]
    cumulative_rewards_multi = [np.mean(reward_avg_multi[:i + 1]) for i in range(start, len(reward_avg_multi))]
    #cumulative_rewards_multi = [np.mean((reward_avg_multi[i - 100:i + 1])) for i in range(100, len(reward_avg_multi))]
    cumulative_rewards_FP = [np.mean(reward_avg_FP[:i + 1]) for i in range(start, len(reward_avg_FP))]
    cumulative_rewards_multi_MIMO = [np.mean(reward_avg_multi_MIMO[:i + 1]) for i in range(start, len(reward_avg_multi_MIMO))]

    cumulative_rewards_optimal = [np.mean(reward_avg_optimal[:i + 1]) for i in range(start, len(reward_avg_optimal))]
    cumulative_rewards_optimal_no_delay = [np.mean(reward_avg_optimal_no_delay[:i + 1]) for i in range(start, len(reward_avg_optimal_no_delay))]
    cumulative_rewards_full_pwr = [np.mean(reward_avg_full_pwr[:i + 1]) for i in
                                   range(start, len(reward_avg_full_pwr))]
    cumulative_rewards_random_pwr = [np.mean(reward_avg_random_pwr[:i + 1]) for i in
                                   range(start, len(reward_avg_full_pwr))]
    plt.subplot(1, 1, 1)


    #plt.plot(range(start, len(reward_avg)), cumulative_rewards, label='Centralized DRL')
    plt.plot(range(start, len(reward_avg_multi)), cumulative_rewards_multi, label='Multi-agent DRL')
    #plt.plot(range(start, len(reward_avg_multi_MIMO)), cumulative_rewards_multi_MIMO, label='Multi-agent DRL MIMO')
    #plt.plot(range(100, len(reward_avg_multi)), cumulative_rewards_multi, label='Multi-agent DRL')


    plt.plot(range(start, len(reward_avg_optimal_no_delay)), cumulative_rewards_optimal_no_delay, label='OPT (no delay)')
    plt.plot(range(start, len(reward_avg_optimal)), cumulative_rewards_optimal, label='OPT (delay)')
    plt.plot(range(start, len(reward_avg_full_pwr)), cumulative_rewards_full_pwr, label='full power')
    plt.plot(range(start, len(reward_avg_random_pwr)), cumulative_rewards_random_pwr, label='random power')
    #plt.plot(range(start, len(reward_avg_FP)), cumulative_rewards_FP, label='FP')
    #plt.plot(range(0, len(reward_avg)), reward_avg, label='Centralized DRL')

    plt.legend()
    plt.xlabel('Time slot')
    plt.ylabel('Moving average of reward')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.show()

    print(np.max(cumulative_rewards), np.max(cumulative_rewards_multi), np.max(cumulative_rewards_FP))


def bitcheck():
    is_64bit = sys.maxsize>2**32
    print(is_64bit)

def testing():
    import os
    #os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin")
    #os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/libnvvp")
    print("All devices:", tf.config.list_physical_devices())
    print("CUDA Available: ", tf.test.is_built_with_cuda())
    print("cuDNN Version: ", tf.test.gpu_device_name())
    print(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))




if __name__=="__main__": ##인터프리터에서 실행할 때만 위 함수(main())을 실행해라. 즉, 다른데서 이 파일을 참조할 땐(import 시) def만 가져가고, 실행은 하지말라는 의미.
    #bitcheck()
    #main()
    main_multi()
    #main_multi_MIMO()
    #opt()
    #fractional()
    #full_pwr()
    graph()
    #testing()