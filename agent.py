import numpy as np


class ValueIteration():
    def __init__(self, states_n, actions_n, P, gamma):
        self.states_n = states_n
        self.actions_n = actions_n
        self.P = P
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.values = np.zeros(self.states_n)
        self.policy = np.zeros(self.states_n)

    def get_action(self, state):
        return int(self.policy[state])

    def render(self):
        print("Values: {}, Policy: {}".format(self.values, self.policy))

    def solve(self, policy_evaluations, iterations: int, delta: float, method: str):
        if method == 'valiter':
            for _ in range(iterations):
                for s in range(self.states_n):
                    values = [sum([prob * (r + self.gamma * self.values[s_])
                                   for prob, s_, r, _ in self.P[s][a]])
                              for a in range(self.actions_n)]
                    self.values[s] = max(values)
                    self.policy[s] = np.argmax(np.array(values))
        elif method == 'politer':

            for _ in range(iterations):
                optimal_policy_found = True
                # policy evaluation
                for _ in range(policy_evaluations):
                    max_diff = 0
                    for s in range(self.states_n):
                        value = sum([prob * (r + self.gamma * self.values[s_])
                                     for prob, s_, r, _ in self.P[s][self.policy[s]]])

                        max_diff = max(max_diff, abs(self.values[s] - value))

                        self.values[s] = value

                    if max_diff < delta:
                        break

                # policy iteration
                for s in range(self.states_n):
                    for a in range(self.actions_n):
                        value = sum([prob * (r + self.gamma * self.values[s_])
                                     for prob, s_, r, _ in self.P[s][a]])

                        if value > self.values[s] and self.policy[s] != a:
                            self.policy[s] = a
                            self.values[s] = value
                            optimal_policy_found = False

                if optimal_policy_found:
                    break
