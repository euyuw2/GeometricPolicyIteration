import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime



# MDP settings

#***
# num_states = 2
# num_actions = 2
# r = np.array([[-0.45, -0.1], [0.5, 0.5]]) # |S| * |A|
# P = np.zeros([num_states, num_actions, num_states])
#
# P[0] = [[0.7, 0.3], # s1, a1
#         [0.99, 0.01]] # s1, a2
# P[1] = [[0.2, 0.8], # s2, a1
#         [0.99, 0.01]] # s2, a2
#***

#***
# num_states = 2
# num_actions = 2
# r = np.array([[0.88, -0.02], [-0.98, 0.42]]) # |S| * |A|
# P = np.zeros([num_states, num_actions, num_states])
#
# P[0] = [[0.96, 0.04], # s1, a1
#         [0.19, 0.81]] # s1, a2
# P[1] = [[0.43, 0.57], # s2, a1
#         [0.72, 0.28]] # s2, a2
# NUM_STATES = 2
# NUM_ACTIONS = 2

I = np.eye(2)
gamma = 0.8
#***


def mdp_manual():

    # num_states = 2
    # num_actions = 2
    # r = np.array([[-0.45, -0.1], [0.5, 0.5]]) # |S| * |A|
    # P = np.zeros([num_states, num_actions, num_states])
    #
    # P[0] = [[1.0, 0.0], # s1, a1
    #         [1.0, 0.0]] # s1, a2
    # P[1] = [[0.0, 1.0], # s2, a1
    #         [1.0, 0.0]] # s2, a2

    num_states = 2
    num_actions = 3
    r = np.array([[-0.93, -0.46, 0.63], [0.78, 0.14, 0.41]]) # |S| * |A|
    P = np.zeros([num_states, num_actions, num_states])

    P[0] = [[0.52, 0.48], # s1, a1
            [0.5, 0.5],
            [0.99, 0.01]] # s1, a3
    P[1] = [[0.85, 0.15], # s2, a1
            [0.11, 0.89],
            [0.1, 0.9]] # s2, a3

    # num_states = 2
    # num_actions = 3
    # r = np.array([[-0.1, -1, 0.1], [0.4, 1.5, 0.1]])  # |S| * |A|
    # P = np.zeros([num_states, num_actions, num_states])
    #
    # P[0] = [[0.9, 0.1],  # s1, a1
    #         [0.2, 0.8],
    #         [0.7, 0.3]]  # s1, a3
    # P[1] = [[0.05, 0.95],  # s2, a1
    #         [0.25, 0.75],
    #         [0.3, 0.7]]  # s2, a3

    # num_states = 2
    # num_actions = 3
    # r = np.array([[-0.1, -1, 0.1], [0.4, 1.5, 0.1]])  # |S| * |A|
    # P = np.zeros([num_states, num_actions, num_states])
    #
    # P[0] = [[0, 1],  # s1, a1
    #         [0, 1],
    #         [1, 0]]  # s1, a3
    # P[1] = [[0, 1],  # s2, a1
    #         [0, 1],
    #         [1, 0]]  # s2, a3

    return r, P


def mdp_gen_2s2a():
    reward = np.random.random([2, 2]) * 2 - 1
    transition = np.random.random([4, 1])
    transition = np.concatenate((transition, 1 - transition), axis=-1)
    transition = np.reshape(transition, [2, 2, 2])
    save_mdp(reward=r, transition=P, num_actions=2, save_path='saved_mdp/')
    return reward, transition


def mdp_gen_2sNa(num_actions):
    reward = np.random.random([2, num_actions]) * 2 - 1
    transition = np.random.random([2 * num_actions, 1])
    transition = np.concatenate((transition, 1 - transition), axis=-1)
    transition = np.reshape(transition, [2, num_actions, 2])
    # save_mdp(reward=reward, transition=transition, num_actions=num_actions, save_path='saved_mdp/')
    return reward, transition


def random_prob_vec(N):
    prob = np.zeros([N])
    perm = np.random.permutation(N)
    scalar = 1
    for i in range(N):
        if i == N - 1:
            prob[perm[i]] = scalar
            return prob
        p = np.random.random() * scalar
        prob[perm[i]] = p
        scalar = 1 - np.sum(prob)


def mdp_gen_NsNa(num_states, num_actions, save=False):
    reward = np.random.random([num_states, num_actions]) * 2 - 1
    transition = np.zeros([num_states, num_actions, num_states], dtype=np.float16)
    for i in range(num_states):
        print('%d / %d' % (i, num_states))
        for j in range(num_actions):
            prob_vec = random_prob_vec(num_states)
            transition[i, j, :] = prob_vec
    if save:
        # save_mdp(reward=reward, transition=transition, save_path='saved_mdp/')
        save_mdp(reward=reward, transition=transition, num_actions=num_actions, save_path='saved_mdp/')
        print('MDP saved')
    return reward, transition


def save_mdp(reward, transition, num_actions, save_path):
    fname = str(datetime.now()) + '_na' + str(num_actions)
    np.save(save_path + fname + '_reward.npy', reward)
    np.save(save_path + fname + '_transition.npy', transition)


def load_mdp(load_path):
    reward = np.load(load_path + '_reward.npy')
    transition = np.load(load_path + '_transition.npy')
    return reward, transition


def func_ub(v1, P, r, id_action):
    return ((1 - gamma * P[0, id_action, 0]) * v1 - r[0, id_action]) / (gamma * P[0, id_action, 1])


def func_lb(v1, P, r, id_action):
    return (r[1, id_action] + gamma * P[1, id_action, 0] * v1) / (1 - gamma * P[1, id_action, 1])


# def func_ub_s1_a1(v1, P, r):
#     return ((1 - gamma * P[0, 0, 0]) * v1 - r[0, 0]) / (gamma * P[0, 0, 1])
#
#
# def func_ub_s1_a2(v1, P, r):
#     return ((1 - gamma * P[0, 1, 0]) * v1 - r[0, 1]) / (gamma * P[0, 1, 1])
#
#
# def func_ub_s1_a3(v1, P, r):
#     return ((1 - gamma * P[0, 2, 0]) * v1 - r[0, 2]) / (gamma * P[0, 2, 1])
#
#
# def func_lb_s2_a1(v1, P, r):
#     return (r[1, 0] + gamma * P[1, 0, 0] * v1) / (1 - gamma * P[1, 0, 1])
#
#
# def func_lb_s2_a2(v1, P, r):
#     return (r[1, 1] + gamma * P[1, 1, 0] * v1) / (1 - gamma * P[1, 1, 1])
#
#
# def func_lb_s2_a3(v1, P, r):
#     return (r[1, 2] + gamma * P[1, 2, 0] * v1) / (1 - gamma * P[1, 2, 1])


def value_func_v2(pi, P, r, num_states, num_actions):
    P_pi = np.zeros([num_states, num_states])
    for s in range(num_states):
        vec = np.sum(np.expand_dims(pi[s], axis=1) * P[s], axis=0)
        P_pi[s] = vec

    r_pi = np.sum(pi * r, axis=-1)
    v = np.matmul(np.linalg.inv(I - gamma * P_pi), r_pi)
    return v


def value_func(pi, P, r, num_states, num_actions):
    P_pi = np.zeros([num_states, num_states])
    for j in range(num_states):
        for k in range(num_states):
            sum_temp = 0
            for i in range(num_actions):
                sum_temp += P[j, i, k] * pi[j, i]
            P_pi[j, k] = sum_temp

    r_pi = np.sum(pi * r, axis=-1)
    v = np.matmul(np.linalg.inv(I - gamma * P_pi), r_pi)
    return v


def random_N_action_policy(num_actions):
    prob = np.zeros([num_actions])
    perm = np.random.permutation(num_actions)
    scalar = 1
    for i in range(num_actions):
        if i == num_actions - 1:
            prob[perm[i]] = scalar
            return prob
        p = np.random.random() * scalar
        prob[perm[i]] = p
        scalar = 1 - np.sum(prob)


def three_action_policy():
    p1 = np.random.random()
    p2 = np.random.random() * (1 - p1)
    p3 = 1 - p1 - p2
    return np.array([p1, p2, p3])



def visualize_mdp(r, P, pi_list=None):

    # r, P = mdp_gen_2s2a()
    num_states, num_actions = r.shape[0], r.shape[1]
    pi = np.zeros([50000, num_states, num_actions]) # n * |S| * |A|

    for i in range(50000):

        probs = random_N_action_policy(num_actions=num_actions)
        pi[i, 0, :] = probs
        probs = random_N_action_policy(num_actions=num_actions)
        pi[i, 1, :] = probs


    pi_d = np.zeros([num_actions * num_actions, 2, num_actions])
    for i in range(num_actions):
        for j in range(num_actions):
            temp = np.zeros([2, num_actions])
            temp[0, i] = 1
            temp[1, j] = 1
            pi_d[i * num_actions + j] = temp


    v_list = []
    for i in range(50000):
        # v = value_func(pi[i], P, r, num_states, num_actions)
        v = value_func_v2(pi[i], P, r, num_states, num_actions)
        v_list.append(v)


    # print('plotting')
    v_samples = np.array(v_list)
    plt.scatter(v_samples[:, 0], v_samples[:, 1], s=2)

    v_list = []
    for i in range(len(pi_d)):
        v = value_func(pi_d[i], P, r, num_states, num_actions)
        v_list.append(v)

    v = np.array(v_list)
    plt.scatter(v[:, 0], v[:, 1], s=20, c='r')
    for i in range(len(pi_d)):
        txt = str(np.argmax(pi_d[i], axis=1))
        plt.text(v[i, 0], v[i, 1], s=txt)

    # plot pi sequence
    v_list = []
    if pi_list is not None:
        pi_d = pi_list
        for i in range(len(pi_d)):
            v = value_func(pi_d[i], P, r, num_states, num_actions)
            v_list.append(v)

        # print('plotting')
        v = np.array(v_list)
        # plt.scatter(v[:, 0], v[:, 1], s=5, c='r')
        for i in range(len(pi_d)):
            if i == 0:
                plt.scatter(v[i, 0], v[i, 1], s=20, c='red')
            elif i == len(pi_d) - 1:
                plt.scatter(v[i, 0], v[i, 1], s=20, c='green')
            else:
                plt.scatter(v[i, 0], v[i, 1], s=10, c='blue')

    # plt.show()
    return v_samples



def visualize_simplex(v, P, r):

    color_ub_list = ['cyan', 'blue', 'purple', 'brown']
    color_lb_list = ['red', 'yellow', 'green', 'orange']

    num_actions = r.shape[1]

    print('plotting LP polytope')
    v1_values = np.arange(np.min(v[:, 0]) - 5, np.max(v[:, 0]) + 5, 0.1)
    v2_ub_list = []
    v2_lb_list = []
    for i in range(num_actions):
        v2_ub_list.append([func_ub(v1, P, r, i) for v1 in v1_values])
        v2_lb_list.append([func_lb(v1, P, r, i) for v1 in v1_values])

    plt.plot(v1_values, [func_lb(v1, P, r, 1) for v1 in v1_values], c='g')

    for i in range(num_actions):
        plt.plot(v1_values, v2_lb_list[i], c=color_lb_list[i])

    for i in range(num_actions):
        plt.plot(v1_values, v2_ub_list[i], c=color_ub_list[i])


    plt.xlim(np.min(v[:, 0]) - 1, np.max(v[:, 0]) + 1)
    plt.ylim(np.min(v[:, 1]) - 1, np.max(v[:, 1]) + 1)
    # plt.legend()
    plt.show()





if __name__ == '__main__':
    # r, P = load_mdp('saved_mdp/2021-07-28 19:04:26.700565_na2')

    # pi_list = np.load('saved_mdp/pi_list.npy')
    # r, P = mdp_manual()
    # r, P = mdp_gen_2sNa(num_actions=2)
    # print('reward:')
    # print(r)
    # print('transition:')
    # print(P)

    r, P = mdp_gen_NsNa(num_states=2, num_actions=2, save=False)
    v = visualize_mdp(r, P, None)
    visualize_simplex(v, P, r)

