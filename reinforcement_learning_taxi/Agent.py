import numpy as np
from collections import defaultdict
from collections import Counter
import math

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.6
        self.eps = 1
        self.gamma = 0.75
        self.rho = 1.5

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.eps = (1 / (i_episode + 1))
        
        
        #GLIE policy
        q = Counter(self.Q[state])
        n_max = q[max(q)]
        n_all = len(list(q.elements()))
        n_non_max = n_all - n_max
        probs = [1 / n_all if n_max == n_all\
                 else (1 - self.eps) / n_max if i == max(q)\
                 else self.eps / n_non_max\
                 for i in q.elements()]
        
        return np.random.choice(a = self.nA, p = probs)

    def step(self, state, action, reward, next_state, done, i_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.alpha = min(1, (1 / (i_episode + 1)) + 0.6)
        
        self.Q[state][action] += self.alpha * (reward + (self.gamma * max(self.Q[next_state])) - (self.Q[state][action] * self.rho))
        