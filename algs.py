import numpy as np
from datetime import datetime
from numpy.linalg import multi_dot
from numpy import expand_dims, matmul, dot
import time
from numba import jit, njit
import os


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


def gen_det_policy_manual(num_actions, vec):
    # vec = np.random.randint(0, num_actions, num_states)
    pi = one_hot(vec, num_actions)
    return pi


def value_func(pi, P, r, gamma):
    num_states, num_actions = P.shape[0], P.shape[1]
    I = np.eye(num_states)

    P_pi = np.zeros([num_states, num_states])
    for s in range(num_states):
        vec = np.sum(np.expand_dims(pi[s], axis=1) * P[s], axis=0)
        P_pi[s] = vec

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


def mdp_manual():
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


def normalize(logits):
    logits_exp = np.exp(logits)
    return logits_exp / np.sum(logits_exp)


def normalize_v2(logits):
    # logits_exp = np.exp(logits)
    return logits / np.sum(logits)



def mdp_gen_NsNa_v2(num_states, num_actions, save=False):
    reward = np.random.random([num_states, num_actions]) * 2 - 1
    transition = np.zeros([num_states, num_actions, num_states], dtype=np.float16)
    for i in range(num_states):
        print('%d / %d' % (i, num_states))
        for j in range(num_actions):
            logits = np.random.random([num_states]) * 10
            prob_vec = normalize_v2(logits)
            # a = np.sum(prob_vec)
            transition[i, j, :] = prob_vec
    if save:
        save_mdp(reward=reward, transition=transition, save_path='saved_mdp/')
        print('MDP saved')
    return reward, transition


def mdp_gen_NsNa(num_states, num_actions, save=False):
    reward = np.random.random([num_states, num_actions]) * 0.1 - 1
    transition = np.zeros([num_states, num_actions, num_states], dtype=np.float16)
    print('generating random MDP')
    for i in range(num_states):
        for j in range(num_actions):
            prob_vec = random_prob_vec(num_states)
            transition[i, j, :] = prob_vec
    if save:
        save_mdp(reward=reward, transition=transition, save_path='saved_mdp/')
        print('MDP saved')
    return reward, transition


def save_mdp(reward, transition, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    num_states, num_actions = reward.shape[0], reward.shape[1]
    # fname = str(time.time()) + '_ns' + str(num_states) + '_na' + str(num_actions)
    # np.save(save_path + fname + '_reward.npy', reward)
    # np.save(save_path + fname + '_transition.npy', transition)
    np.save(save_path + 'ns' + str(num_states) + '_na' + str(num_actions) + '_reward.npy', reward)
    np.save(save_path + 'ns' + str(num_states) + '_na' + str(num_actions) + '_transition.npy', transition)


def bellman_operator(P, r, gamma, v):
    v_max = np.zeros(len(v))
    for s in range(len(v)):
        v_max[s] = max(r[s, a] + gamma * np.matmul(P[s, a, :], v) for a in range(r.shape[1]))
    return v_max


def vec_equal(x, y, eps=1e-4):
    return np.sum((x-y) ** 2) < eps


def is_optimal(P, r, gamma, v):
    v_max = bellman_operator(P, r, gamma, v)
    return vec_equal(v, v_max)


def simplex_pi_update(v, P, r, gamma, pi):
    """

    :param v:
    :param P:
    :param r:
    :param gamma:
    :param pi:
    :return: the updated policy
    """
    eps = 1e-8
    num_states, num_actions = P.shape[0], P.shape[1]
    best_state, best_action, best_adv = None, None, eps
    for s in range(num_states):
        for a in range(num_actions):
            # temp = value_func(pi_temp, P, r, gamma) - v[s]
            adv = r[s, a] + gamma * np.matmul(P[s, a, :], v) - v[s]
            # print(s, a, adv)
            if adv > best_adv:
                best_adv, best_state, best_action = adv, s, a

    return best_state, best_action


def is_policy_same(a, b):
    sum = np.sum(np.abs(a - b) ** 2)
    return sum == 0


def pi_update_v(v, P, Q, gamma, pi, num_switches):
    """

    :param v:
    :param P:
    :param r:
    :param gamma:
    :param pi:
    :return: the updated policy
    """
    eps = 1e-8
    pi_new = np.copy(pi)
    num_states, num_actions = P.shape[0], P.shape[1]

    for s in range(num_states):
        a_old = np.argmax(pi[s])
        one_s = one_hot_s(num_states, s)
        v_adv_list = [vs_update(P, Q, s, s, a_old, a, gamma, v, one_s) - v[s] for a in range(num_actions)]
        a_best = np.argmax(v_adv_list)
        best_adv = v_adv_list[a_best]
        if best_adv > eps:
            num_switches += 1
            vec = np.zeros([num_actions])
            vec[a_best] = 1
            pi_new[s] = vec
    return pi_new, num_switches



def pi_update(v, P, r, gamma, pi, num_switches):
    """

    :param v:
    :param P:
    :param r:
    :param gamma:
    :param pi:
    :return: the updated policy
    """
    eps = 1e-8
    pi_new = np.copy(pi)
    num_states, num_actions = P.shape[0], P.shape[1]

    for s in range(num_states):
        best_action, best_adv = None, eps
        for a in range(num_actions):
            # temp = value_func(pi_temp, P, r, gamma) - v[s]
            adv = r[s, a] + gamma * np.matmul(P[s, a, :], v) - v[s]
            # print(s, a, adv)
            if adv > best_adv:
                best_adv, best_action = adv, a
        if best_adv > eps:
            num_switches += 1
            vec = np.zeros([num_actions])
            vec[best_action] = 1
            pi_new[s] = vec
    return pi_new, num_switches



def policy_iteration_v(P, r, gamma, pi):
    # num_states, num_actions = r.shape[0], r.shape[1]
    # pi = gen_det_policy(num_states, num_actions).astype(np.uint8)
    # pi = gen_det_policy_manual(num_actions, [0, 1]).astype(np.uint8)
    # v = value_func(pi, P, r, gamma)
    v, Q, r_pi = value_func_v3(pi, P, r, gamma)
    iter_no = 1
    num_switches = 0
    while 1:
        print('policy iteration: no. iter: %d' % iter_no)
        pi_new, num_switches = pi_update_v(v, P, Q, gamma, pi, num_switches)

        v, Q, r_pi = value_func_v3(pi_new, P, r, gamma)
        if is_optimal(P, r, gamma, np.squeeze(v)):
            print('num switches: %d' % num_switches)
            return v, pi_new
        else:
            pi = pi_new
            iter_no += 1


def policy_iteration(P, r, gamma, pi):
    # num_states, num_actions = r.shape[0], r.shape[1]
    # pi = gen_det_policy(num_states, num_actions).astype(np.uint8)
    # pi = gen_det_policy_manual(num_actions, [0, 1]).astype(np.uint8)
    v = value_func(pi, P, r, gamma)

    iter_no = 1
    pi_list = [pi]
    num_switches = 0
    while 1:
        print('PI : no. iter: %d' % iter_no)
        pi_new, num_switches = pi_update(v, P, r, gamma, pi, num_switches)

        # eps = 1e-8
        # pi_new = np.copy(pi)
        # num_states, num_actions = P.shape[0], P.shape[1]
        #
        # for s in range(num_states):
        #     best_action, best_adv = None, eps
        #     for a in range(num_actions):
        #         # temp = value_func(pi_temp, P, r, gamma) - v[s]
        #         adv = r[s, a] + gamma * np.matmul(P[s, a, :], v) - v[s]
        #         # print(s, a, adv)
        #         if adv > best_adv:
        #             best_adv, best_action = adv, a
        #     if best_adv > eps:
        #         num_switches += 1
        #         vec = np.zeros([num_actions])
        #         vec[best_action] = 1
        #         pi_new[s] = vec

        v = value_func(pi_new, P, r, gamma)
        if is_optimal(P, r, gamma, v):
            print('num switches: %d' % num_switches)
            return v, pi_new, iter_no, num_switches
        else:
            pi_list.append(pi_new)
            pi = pi_new
            iter_no += 1


def simplex_policy_iteration(P, r, gamma, pi):
    P, r = P.astype(np.float64), r.astype(np.float64)
    num_states, num_actions = r.shape[0], r.shape[1]
    v, Q, r_pi = value_func_v3(pi, P, r, gamma)
    iter_no = 1
    # count = {}
    # for i in range(num_states):
    #     count[i] = 0
    while 1:
        print('simple policy iteration no. iter: %d' % iter_no)
        s, a = simplex_pi_update(v, P, r, gamma, pi)
        # count[s] += 1
        # print(count)
        a_old = np.argmax(pi[s])
        v, Q, r_pi = v_update_sherman_morrison(P, Q, r_pi, s, a_old, a, gamma)
        if is_optimal(P, r, gamma, np.squeeze(v)):
            vec = np.zeros([num_actions])
            vec[a] = 1
            pi[s] = vec
            iter_no += 1
            return v, pi, iter_no
        else:
            vec = np.zeros([num_actions])
            vec[a] = 1
            pi[s] = vec
            iter_no += 1


def load_mdp(load_path):
    reward = np.load(load_path + '_reward.npy')
    transition = np.load(load_path + '_transition.npy')
    return reward, transition


def value_func_v3(pi, P, r, gamma):
    num_states, num_actions = P.shape[0], P.shape[1]
    I = np.eye(num_states)

    P_pi = np.zeros([num_states, num_states])
    for s in range(num_states):
        vec = np.sum(np.expand_dims(pi[s], axis=1) * P[s], axis=0)
        P_pi[s] = vec

    r_pi = np.sum(pi * r, axis=-1, keepdims=True)
    Q = np.linalg.inv(I - gamma * P_pi)
    v = np.matmul(Q, r_pi)
    return v, Q, r_pi


def one_hot_s(num_states, s):
    one_s = np.zeros([1, num_states])
    one_s[0, s] = 1
    return one_s


def vs_update(P, Q, s, s_p, a_old, a, gamma, v, one_s_p):
    q_s = expand_dims(Q[:, s], axis=1)
    w = gamma * np.expand_dims(P[s, a, :] - P[s, a_old, :], axis=0)
    scalar = Q[s_p, s] / (1 - np.matmul(w, q_s))
    temp = (1 - np.matmul(w, q_s))
    v_s = matmul((one_s_p + scalar * w), v + (r[s, a] - r[s, a_old]) * q_s)
    return v_s


def v_update_sherman_morrison(P, Q, r_pi, s, a_old, a, gamma):
    q_s = expand_dims(Q[:, s], axis=1)
    w = gamma * expand_dims(P[s, a, :] - P[s, a_old, :], axis=0)
    Q_new = Q + dot(q_s, dot(w, Q)) / (1.0 - matmul(w, q_s))
    r_pi[s] = r[s, a]
    v = matmul(Q_new, r_pi)
    return v, Q_new, r_pi


def policy_update_single_state(pi, s, a):
    vec = np.zeros([pi.shape[1]])
    vec[a] = 1
    pi[s] = vec
    return pi


@njit(cache=True)
def geometric_policy_iteration_update(s,
                                      num_actions,
                                      v,
                                      P_s,
                                      Q_T,
                                      r_s,
                                      r_pi,
                                      eps,
                                      num_switches,
                                      a_old,
                                      one_s,
                                      pi,
                                      gamma):


    a_s, best_adv = None, np.array([eps])
    q_s = expand_dims(Q_T[s, :], axis=1)
    for a in range(num_actions):
        w = gamma * np.expand_dims(P_s[a] - P_s[a_old], axis=0)
        scalar = Q_T[s, s] / (1 - dot(w, q_s))
        v_s = dot(one_s + np.multiply(scalar, w), v + (r_s[a] - r_s[a_old]) * q_s)
        adv = np.reshape(v_s, -1) - np.reshape(v[s], -1)
        if adv > best_adv:
            best_adv = adv
            a_s = a
    if best_adv > eps:
        a_s = int(a_s)
        num_switches += 1
        q_s_T = np.transpose(q_s)
        w = gamma * expand_dims(P_s[a_old] - P_s[a_s], axis=1)
        Q_T = Q_T - dot(dot(Q_T, w), q_s_T) / (1.0 + dot(q_s_T, w))
        r_pi[s] = r_s[a_s]
        v = dot(np.transpose(Q_T), r_pi)
        pi[s] = a_s
    return Q_T, v, pi, r_pi, num_switches


def gpi_v(P, Q_T, r_pi, v, r, gamma, pi):
    num_states, num_actions = r.shape[0], r.shape[1]
    num_iter = 1
    eps = 1e-8
    num_switches = 0
    while 1:
        print('GPI-V iter no: %d' % num_iter)
        for s in range(num_states):
            one_s = np.zeros((1, num_states))
            one_s[0, s] = 1
            P_s, r_s = P[s], r[s]
            a_old = int(pi[s])
            Q_T, v, pi, r_pi, num_switches = geometric_policy_iteration_update(s,
                                                                              num_actions,
                                                                              v,
                                                                              P_s,
                                                                              Q_T,
                                                                              r_s,
                                                                              r_pi,
                                                                              eps,
                                                                              num_switches,
                                                                              a_old,
                                                                              one_s,
                                                                              pi,
                                                                              gamma)
        if is_optimal(P, r, gamma, np.reshape(v, -1)):
            return v, pi, num_iter, num_switches
        num_iter += 1


# def geometric_policy_iteration_plus(P, r, gamma, pi):
#     P, r = np.float64(P), np.float64(r)
#     num_states, num_actions = r.shape[0], r.shape[1]
#     v, Q, r_pi = value_func_v3(pi, P, r, gamma)
#     iter_no = 1
#     eps = 1e-8
#     num_switches = 0
#     while 1:
#         print('fast policy iteration iter no. %d' % iter_no)
#         pi_new = np.copy(pi)
#         # print('state: ', end=' ')
#         for s in range(num_states):
#             # print(s, end=' ')
#             a_old = np.argmax(pi[s, :])
#             one_s = one_hot_s(num_states, s)
#             v_adv_list = [vs_update(P, Q, s, s, a_old, a, gamma, v, one_s) - v[s] for a in range(num_actions)]
#             a_selected = np.argmax(v_adv_list)
#             if v_adv_list[a_selected] > eps:
#                 num_switches += 1
#                 v_new, Q, r_pi = v_update_sherman_morrison(P, Q, r_pi, s, a_old, a_selected, gamma)
#                 v = v_new
#                 pi_new = policy_update_single_state(pi_new, s, a_selected)
#
#         pi = pi_new
#
#         if is_optimal(P, r, gamma, np.squeeze(v)):
#             print('num switches: %d' % num_switches)
#             return v, pi, iter_no, num_switches
#
#         iter_no += 1


def geometric_policy_iteration(P, r, gamma, pi):
    P, r = np.float32(P), np.float32(r)
    num_states, num_actions = r.shape[0], r.shape[1]
    v, Q, r_pi = value_func_v3(pi, P, r, gamma)
    iter_no = 1
    eps = 1e-8
    num_switches = 0
    while 1:
        print('GPI iter no. %d' % iter_no)
        pi_new = np.copy(pi)
        # print('state: ', end=' ')
        for s in range(num_states):
            # print(s, end=' ')
            a_old = np.argmax(pi[s, :])
            # one_s = one_hot_s(num_states, s)
            # v_adv_list = [vs_update(P, Q, s, s, a_old, a, gamma, v, one_s) - v[s] for a in range(num_actions)]
            adv_list = [r[s, a] + gamma * np.matmul(P[s, a, :], v) - v[s] for a in range(num_actions)]
            a_selected = np.argmax(adv_list)
            # print(v_adv_list[a_selected])
            if adv_list[a_selected] > eps:
                num_switches += 1
                v, Q, r_pi = v_update_sherman_morrison(P, Q, r_pi, s, a_old, a_selected, gamma)
                pi_new = policy_update_single_state(pi_new, s, a_selected)

        pi = pi_new

        if is_optimal(P, r, gamma, np.squeeze(v)):
            print('num switches: %d' % num_switches)
            return v, pi, iter_no, num_switches

        iter_no += 1


# def fast_policy_iteration_test(P, r, gamma, pi):
#     P, r = np.float64(P), np.float64(r)
#     num_states, num_actions = r.shape[0], r.shape[1]
#     v, Q, r_pi = value_func_v3(pi, P, r, gamma)
#     iter_no = 1
#     eps = 1e-8
#     num_switches = 0
#     pi = np.argmax(pi, axis=1)
#     while 1:
#         print('fast policy iteration iter no. %d' % iter_no)
#         # print('state: ', end=' ')
#         for s in range(num_states):
#             # print(s, end=' ')
#             a_old = pi[s]
#             one_s = one_hot_s(num_states, s)
#             v_adv_list = [vs_update(P, Q, s, s, a_old, a, gamma, v, one_s) - v[s] for a in range(num_actions)]
#             a_selected = np.argmax(v_adv_list)
#             if v_adv_list[a_selected] > eps:
#                 num_switches += 1
#                 v, Q, r_pi = v_update_sherman_morrison(P, Q, r_pi, s, a_old, a_selected, gamma)
#                 # q_s = expand_dims(Q[:, s], axis=1)
#                 # w = gamma * expand_dims(P[s, a_old, :] - P[s, a_selected, :], axis=0)
#                 # Q = Q - multi_dot([q_s, w, Q]) / (1.0 + matmul(w, q_s))
#                 # r_pi[s] = r[s, a_selected]
#                 # v = matmul(Q, r_pi)
#                 pi[s] = a_selected
#
#         if is_optimal(P, r, gamma, np.squeeze(v)):
#             print('num switches: %d' % num_switches)
#             return v, pi
#
#         iter_no += 1


if __name__ == '__main__':

    r, P = mdp_gen_NsNa(num_states=100, num_actions=100, save=True)

    # pi = gen_det_policy(num_states=r.shape[0], num_actions=r.shape[1])
    # np.save('saved_mdp/pi_initial.npy', pi)

    # r, P = load_mdp('saved_mdp/ns300_na600')

    r, P = r.astype(np.float32), P.astype(np.float32)


    spi_list, pi_list, gpi_list, gpiv_list = [], [], [], []

    rounds = 2
    for i in range(rounds + 1):
        print(i)
        pi = gen_det_policy(num_states=r.shape[0], num_actions=r.shape[1])

        if i > 0:
            start_time = time.time()
            v_spi, pi_spi, iter_spi = simplex_policy_iteration(P=P, r=r, gamma=0.9, pi=np.copy(pi))
            time_spi = time.time() - start_time
            spi_list.append([time_spi, iter_spi])
            print('%.4f sec' % (time.time() - start_time))

        if i > 0:
            start_time = time.time()
            v_pi, pi_pi, iter_pi, num_switches_pi = policy_iteration(P=P, r=r, gamma=0.9, pi=np.copy(pi))
            time_pi = time.time() - start_time
            pi_list.append([time_pi, iter_pi, num_switches_pi])
            print('%.4f sec' % (time.time() - start_time))

        # start_time = time.time()
        # v_piv, pi_piv = policy_iteration_v(P=P, r=r, gamma=0.9, pi=np.copy(pi))
        # print('%.4f sec' % (time.time() - start_time))

            start_time = time.time()
            v_gpi, pi_gpi, iter_gpi, num_switches_gpi = geometric_policy_iteration(P=P, r=r, gamma=0.9,
                                                                                      pi=np.copy(pi))
            time_gpi = time.time() - start_time
            if i > 0:
                gpi_list.append([time_gpi, iter_gpi, num_switches_gpi])
            print('%.4f sec' % (time.time() - start_time))

        v, Q, r_pi = value_func_v3(pi, P, r, 0.9)
        pi = np.argmax(pi, axis=1)
        start_time = time.time()
        v_gpiv, pi_gpiv, iter_gpi, num_switches_gpi = gpi_v(P=P, Q_T=Q.transpose(), r=r, r_pi=r_pi, v=v, gamma=0.9, pi=pi)
        time_gpiv = time.time() - start_time
        if i > 0:
            gpiv_list.append([time_gpiv, iter_gpi, num_switches_gpi])
        print('%.4f sec' % (time.time() - start_time))

    print('spi')
    print(np.mean(np.array(spi_list), axis=0))
    print('pi')
    print(np.mean(np.array(pi_list), axis=0))
    print('gpi')
    print(np.mean(np.array(gpi_list), axis=0))
    print('gpiv')
    print(np.mean(np.array(gpiv_list), axis=0))