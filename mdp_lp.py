import cvxpy as  cp
import numpy as np

# MDP settings

def load_mdp(load_path):
    reward = np.load(load_path + '_reward.npy')
    transition = np.load(load_path + '_transition.npy')
    return reward, transition

num_states = 3
num_actions = 2
# r = np.array([[-0.45, -0.1], [0.5, 0.5]]) # |S| * |A|
# P = np.zeros([num_states, num_actions, num_states])

# P[0] = [[0.7, 0.3], # s1, a1
#         [0.99, 0.01]] # s1, a2
# P[1] = [[0.2, 0.8], # s2, a1
#         [0.99, 0.01]] # s2, a2

r, P = load_mdp('saved_mdp/2021-07-12 01:43:14.392909_na2')

I = np.eye(num_states)
gamma = 0.9

v = cp.Variable((2, 1), boolean=False)

constraints = []

for i in range(num_states):
    for j in range(num_actions):
        constraints.append(v[i] >= (r[i, j] + gamma * (P[i, j, 0] * v[0] + P[i, j, 1] * v[1])))

obj = cp.Minimize(cp.sum(v) / 2.0)
prob = cp.Problem(objective=obj, constraints=constraints)
obj_solved = prob.solve(verbose=True)
print(obj_solved)
print(v.value)