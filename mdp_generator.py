import numpy as np
from datetime import datetime
import time



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



def save_mdp(reward, transition, save_path):
    num_states, num_actions = reward.shape[0], reward.shape[1]
    # fname = str(time.time()) + '_ns' + str(num_states) + '_na' + str(num_actions)
    # np.save(save_path + fname + '_reward.npy', reward)
    # np.save(save_path + fname + '_transition.npy', transition)
    np.save(save_path + 'ns' + str(num_states) + '_na' + str(num_actions) + '_reward.npy', reward)
    np.save(save_path + 'ns' + str(num_states) + '_na' + str(num_actions) + '_transition.npy', transition)


def mdp_gen_NsNa(num_states, num_actions, save=False):
    reward = np.random.random([num_states, num_actions]) * 2 - 1
    transition = np.zeros([num_states, num_actions, num_states], dtype=np.float16)
    for i in range(num_states):
        print('%d / %d' % (i, num_states))
        for j in range(num_actions):
            prob_vec = random_prob_vec(num_states)
            transition[i, j, :] = prob_vec
    if save:
        save_mdp(reward=reward, transition=transition, save_path='saved_mdp/')
        print('MDP saved')
    # return reward, transition



def one_hot(vec, num_actions):
    """

    :param vec: action index, idx starts from 0 to num_actions-1
    :return: one-hot encoding of pi
    """
    pi = np.zeros([len(vec), num_actions])
    for i in range(len(vec)):
        pi[i, vec[i]] = 1
    return pi


def gen_det_policy(num_states, num_actions):
    vec = np.random.randint(0, num_actions, num_states)
    pi = one_hot(vec, num_actions)
    return pi


def load_mdp(load_path):
    reward = np.load(load_path + '_reward.npy')
    transition = np.load(load_path + '_transition.npy')
    return reward, transition



if __name__ == '__main__':
    # mdp_gen_NsNa(num_states=5000, num_actions=200, save=True)
    r, P = load_mdp('saved_mdp/ns1000_na300')
    pi = gen_det_policy(num_states=r.shape[0], num_actions=r.shape[1])
    np.save('saved_mdp/pi_initial.npy', pi)