import random
import numpy as np
import matplotlib.pyplot as plt


class Gambler:
    def __init__(self, goal=100, prob_h=0.4):
        self.GOAL = goal
        self.PROB_H = prob_h
        self.STATES = np.arange(self.GOAL + 1)

        self.state_values = np.zeros(self.GOAL + 1)
        self.state_values[self.GOAL] = 1.0             # Final state with a reward of +1
        self.sweeps = []

    def value_iteration(self):
        is_converged = False

        while is_converged == False:
            prev_state_value = self.state_values.copy()
            self.sweeps.append(prev_state_value)

            for state in self.STATES[1:self.GOAL]:
                actions = np.arange(min(state, self.GOAL - state) + 1)
                action_returns = []
                for action in actions:
                    action_returns.append(self.PROB_H * self.state_values[state + action] +
                                          (1 - self.PROB_H) * self.state_values[state - action])
                new_value = np.max(action_returns)
                self.state_values[state] = new_value
            if abs(self.state_values - prev_state_value).max() < 1e-6:
                self.sweeps.append(self.state_values)
                is_converged = True

    def calculate_optimal_policy(self):
        policy = np.zeros(self.GOAL + 1)
        for state in self.STATES[1:self.GOAL]:
            actions = np.arange(min(state, self.GOAL - state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(
                    self.PROB_H * self.state_values[state + action] + (1 - self.PROB_H) * self.state_values[state - action])

            # round to resemble the figure in the book, see
            # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
            policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
        return policy

    def plot_sweeps(self):
        plt.figure(figsize=(10, 20))

        plt.subplot(2, 1, 1)
        for sweep, state_value in enumerate(self.sweeps):
            plt.plot(state_value, label='sweep {}'.format(sweep))
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        plt.legend(loc='best')

        plt.savefig('sweeps.png')
        plt.close()

    def plot_optimal_policy(self, policy):
        plt.figure(figsize=(10, 20))

        plt.subplot(2, 1, 2)
        plt.scatter(self.STATES, policy)
        plt.xlabel('Capital')
        plt.ylabel('Final policy (stake)')

        plt.savefig('final_policy.png')
        plt.close()

    def run(self):
        self.value_iteration()
        optimal_policy = self.calculate_optimal_policy()

        self.plot_sweeps()
        self.plot_optimal_policy(optimal_policy)


def main():
    gambler = Gambler()
    gambler.run()


main()
