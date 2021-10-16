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

NUM_STATES = 3
NUM_ACTIONS = 2

I = np.eye(NUM_STATES)
gamma = 0.9
#***

ax = plt.axes(projection='3d')

def mdp_manual():

    # num_states = 2
    # num_actions = 3
    # r = np.array([[-0.93, -0.46, 0.63], [0.78, 0.14, 0.41]]) # |S| * |A|
    # P = np.zeros([num_states, num_actions, num_states])
    #
    # P[0] = [[0.52, 0.48], # s1, a1
    #         [0.5, 0.5],
    #         [0.99, 0.01]] # s1, a3
    # P[1] = [[0.85, 0.15], # s2, a1
    #         [0.11, 0.89],
    #         [0.1, 0.9]] # s2, a3

    num_states = 2
    num_actions = 3
    r = np.array([[-0.1, -1, 0.1], [0.4, 1.5, 0.1]])  # |S| * |A|
    P = np.zeros([num_states, num_actions, num_states])

    P[0] = [[0.9, 0.1],  # s1, a1
            [0.2, 0.8],
            [0.7, 0.3]]  # s1, a3
    P[1] = [[0.05, 0.95],  # s2, a1
            [0.25, 0.75],
            [0.3, 0.7]]  # s2, a3

    return r, P


def mdp_gen_2s2a():
    reward = np.random.random([2, 2]) * 2 - 1
    transition = np.random.random([4, 1])
    transition = np.concatenate((transition, 1 - transition), axis=-1)
    transition = np.reshape(transition, [2, 2, 2])
    save_mdp(reward=r, transition=P, num_actions=2, save_path='saved_mdp/')
    return reward, transition


def mdp_gen_3sNa(num_actions):
    reward = np.random.random([3, num_actions]) * 2 - 1
    transition = np.random.random([3 * num_actions, 1])
    transition = np.concatenate((transition, 1 - transition), axis=-1)
    transition = np.reshape(transition, [3, num_actions, 2])
    # save_mdp(reward=reward, transition=transition, num_actions=num_actions, save_path='saved_mdp/')
    return reward, transition

def mdp_gen_NsNa(save=False):
    reward = np.random.random([NUM_STATES, NUM_ACTIONS]) * 2 - 1
    transition = np.zeros([NUM_STATES, NUM_ACTIONS, NUM_STATES])
    for i in range(NUM_STATES):
        for j in range(NUM_ACTIONS):
            prob_vec = random_prob_vec(NUM_STATES)
            transition[i, j, :] = prob_vec
    # transition = np.random.random([NUM_STATES * NUM_ACTIONS, 1])
    # transition = np.concatenate((transition, 1 - transition), axis=-1)
    # transition = np.reshape(transition, [NUM_STATES, NUM_ACTIONS, 2])
    if save:
        save_mdp(reward=reward, transition=transition, num_actions=NUM_ACTIONS, save_path='saved_mdp/')
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


def func_ub1_3d(v1, v2, P, r, id_action):
    return ((1 - gamma * P[0, id_action, 0]) * v1 - gamma * P[0, id_action, 1] * v2 - r[0, id_action]) / (gamma * P[0, id_action, 2])


def func_ub2_3d(v1, v2, P, r, id_action):
    return ((1 - gamma * P[1, id_action, 1]) * v2 - gamma * P[1, id_action, 0] * v1 - r[1, id_action]) / (gamma * P[1, id_action, 2])


def func_lb_3d(v1, v2, P, r, id_action):
    return (r[2, id_action] + gamma * (P[2, id_action, 0] * v1 + P[2, id_action, 1] * v2)) / (1 - gamma * P[2, id_action, 2])


# def func_ub(v1, P, r, id_action):
#     return ((1 - gamma * P[0, id_action, 0]) * v1 - r[0, id_action]) / (gamma * P[0, id_action, 1])
#
#
# def func_lb(v1, P, r, id_action):
#     return (r[1, id_action] + gamma * P[1, id_action, 0] * v1) / (1 - gamma * P[1, id_action, 1])



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


# def random_N_action_policy(num_actions):
#     prob = np.zeros([num_actions])
#     perm = np.random.permutation(num_actions)
#     scalar = 1
#     for i in range(num_actions):
#         if i == num_actions - 1:
#             prob[perm[i]] = scalar
#             return prob
#         p = np.random.random() * scalar
#         prob[perm[i]] = p
#         scalar = 1 - np.sum(prob)


def three_action_policy():
    p1 = np.random.random()
    p2 = np.random.random() * (1 - p1)
    p3 = 1 - p1 - p2
    return np.array([p1, p2, p3])



def visualize_mdp(r, P):

    # r, P = mdp_gen_2s2a()
    num_states, num_actions = r.shape[0], r.shape[1]
    pi = np.zeros([50000, num_states, num_actions]) # n * |S| * |A|

    for i in range(50000):
        for j in range(num_states):
            probs = random_prob_vec(num_actions)
            pi[i, j, :] = probs

    pi_d = np.zeros([num_actions ** num_states, num_states, num_actions])
    cnt = 0
    for i in range(num_actions):
        for j in range(num_actions):
            for k in range(num_actions):
                temp = np.zeros([num_states, num_actions])
                temp[0, i] = 1
                temp[1, j] = 1
                temp[2, k] = 1
                pi_d[cnt] = temp
                cnt += 1


    v_list = []
    for i in range(5000):
        v = value_func(pi[i], P, r, num_states, num_actions)
        v_list.append(v)


    print('plotting')
    v = np.array(v_list)

    # plt3d = plt.figure().gca(projection='3d')
    # plt.scatter(v[:, 0], v[:, 1], s=2)

    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2], s=2, c='b', alpha=0.2)
    v_list = []
    for i in range(len(pi_d)):
        v = value_func(pi_d[i], P, r, num_states, num_actions)
        v_list.append(v)


    print('plotting')
    v = np.array(v_list)
    for i in range(len(pi_d)):
        ax.scatter3D(v[:, 0], v[:, 1], v[:, 2], s=10, c='r')

    plt.xlabel('v1')
    plt.ylabel('v2')
    # plt.zlabel('v3')
    # plt.set(xlabel='v1', ylabel='v2', zlabel='v3')
    # plt.show()

    return v


def visualize_simplex(v, P, r):
    color_ub_list = ['g', 'b', 'g', 'b', 'pink']
    # color_ub_list = ['black', 'red', 'green', 'blue']
    color_lb_list = ['red', 'yellow', 'green', 'orange']

    num_actions = r.shape[1]

    print('plotting LP polytope')

    v1_values = np.linspace(np.min(v[:, 0]) - 1, np.max(v[:, 0]) + 1, 2)
    v2_values = np.linspace(np.min(v[:, 1]) - 1, np.max(v[:, 1]) + 1, 2)
    X, Y = np.meshgrid(v1_values, v2_values)

    # plt3d = plt.figure().gca(projection='3d')

    for i in range(num_actions):
        Z = func_ub1_3d(X, Y, P, r, i)
        ax.plot_surface(X, Y, Z, alpha=0.2, color=color_ub_list[2 * i])
        # ax.plot_surface(X, Y, Z, alpha=0.2, color='k')

        Z = func_ub2_3d(X, Y, P, r, i)
        ax.plot_surface(X, Y, Z, alpha=0.2, color=color_ub_list[2 * i + 1])

    for i in range(num_actions):
        Z = func_lb_3d(X, Y, P, r, i)
        ax.plot_surface(X, Y, Z, alpha=0.2, color='r')

    # plt.xlim(np.min(v[:, 0]) - 3, np.max(v[:, 0]) + 3)
    # plt.ylim(np.min(v[:, 1]) - 3, np.max(v[:, 1]) + 3)
    ax.set_xlim(np.min(v[:, 0]) - 10, np.max(v[:, 0]) + 10)
    ax.set_ylim(np.min(v[:, 1]) - 10, np.max(v[:, 1]) + 10)
    ax.set_zlim(np.min(v[:, 2]) - 10, np.max(v[:, 2]) + 10)
    ax.set(xlabel='v1', ylabel='v2', zlabel='v3')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # r, P = load_mdp('saved_mdp/2021-07-12 01:43:14.392909_na2')
    # r, P = mdp_manual()
    r, P = mdp_gen_NsNa()
    v = visualize_mdp(r, P)
    visualize_simplex(v, P, r)