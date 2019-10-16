import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import progressbar
from scipy.stats import poisson

matplotlib.use('Agg')


class RentalAgency:
    def __init__(self, max_cars=20, max_move=5, gamma=0.9, car_revenue=10, move_cost=2):
        self.MAX_CARS = max_cars
        self.MAX_MOVE = max_move
        self.GAMMA = gamma
        self.CAR_REVENUE = car_revenue
        self.MOVE_COST = move_cost
        self.FIRST_LOCATION = {'requests': 3, 'returns': 3}
        self.SECOND_LOCATION = {'requests': 4, 'returns': 2}

        self.value = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        self.policy = np.zeros(self.value.shape)
        self.actions = np.arange(-self.MAX_MOVE, self.MAX_MOVE + 1)
        self.poisson_cache = dict()

    def calculate_probability(self, lam, size):
        key = size * 10 + lam
        if key not in self.poisson_cache:
            self.poisson_cache[key] = poisson.pmf(size, lam)
        return self.poisson_cache[key]

    def expected_return(self, state, action):
        UPPER_BOUND = self.MAX_MOVE * 2 + 1

        returns = -self.MOVE_COST * abs(action)

        location_1_cars = int(min(state[0] - action, self.MAX_CARS))
        location_2_cars = int(min(state[1] + action, self.MAX_CARS))

        for rental_request_first_loc in range(UPPER_BOUND):
            for rental_request_second_loc in range(UPPER_BOUND):
                prob_first_request = self.calculate_probability(
                    size=rental_request_first_loc, lam=self.FIRST_LOCATION['requests'])
                prob_second_request = self.calculate_probability(
                    size=rental_request_second_loc, lam=self.SECOND_LOCATION['requests'])
                prob_request = prob_first_request * prob_second_request

                num_of_cars_first_loc = location_1_cars
                num_of_cars_second_loc = location_2_cars

                valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
                valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

                reward = (valid_rental_first_loc + valid_rental_second_loc) * self.CAR_REVENUE
                num_of_cars_first_loc -= valid_rental_first_loc
                num_of_cars_second_loc -= valid_rental_second_loc

                for returned_cars_first_loc in range(UPPER_BOUND):
                    for returned_cars_second_loc in range(UPPER_BOUND):
                        prob_return = self.calculate_probability(size=returned_cars_first_loc, lam=self.FIRST_LOCATION['returns']) * self.calculate_probability(
                            size=returned_cars_second_loc, lam=self.SECOND_LOCATION['returns'])

                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, self.MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, self.MAX_CARS)
                        joint_probability = prob_return * prob_request
                        returns += joint_probability * (reward + self.GAMMA *
                                                        self.value[num_of_cars_first_loc_, num_of_cars_second_loc_])
        return returns

    def evaluate_policy(self, epsilon=1e-4):
        delta = 100
        while delta > epsilon:
            prev_state_value = self.value.copy()

            # bar = progressbar.ProgressBar(maxval=self.MAX_CARS + 1,
            #                               widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            # bar.start()
            for i in range(self.MAX_CARS + 1):
                # bar.update(i)

                for j in range(self.MAX_CARS + 1):
                    # print('i', i, 'j', j)

                    self.value[i, j] = self.expected_return([i, j], self.policy[i, j])
            delta = abs(prev_state_value - self.value).max()
            # bar.finish()
            print('Delta: ', delta)

        return

    def improve_policy(self):
        policy_stable = True
        for i in range(self.MAX_CARS + 1):
            for j in range(self.MAX_CARS + 1):
                prev_action = self.policy[i, j]
                action_returns = []
                for action in self.actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(self.expected_return([i, j], action))
                    else:
                        action_returns.append(-np.inf)
                new_action = self.actions[np.argmax(action_returns)]
                self.policy[i, j] = new_action
                if policy_stable and prev_action != new_action:
                    policy_stable = False
        return policy_stable

    def evaluate(self):
        iterations = 0
        stable_policy = False
        while stable_policy == False and iterations < 100:
            print('Iteration', iterations)
            self.evaluate_policy()
            print('\tDone evaluating policy')
            stable_policy = self.improve_policy()
            print('\tDone improving policy')
            iterations += 1

        return iterations

    def plot(self, iterations,  filename='figure4.2.png'):
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)  # "YlGnBu"
        _, axes = plt.subplots(2, 3, figsize=(40, 20))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()

        fig = sns.heatmap(np.flipud(self.policy), cmap=cmap, ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(self.MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        plt.savefig(filename)
        plt.close()

        return


def main():
    print('Welcome to Mr. Jack\'s rental agency')
    jacks_agency = RentalAgency()
    iterations = jacks_agency.evaluate()
    jacks_agency.plot(iterations=iterations)


main()
