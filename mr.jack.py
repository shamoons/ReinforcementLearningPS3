import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

matplotlib.use('Agg')


class RentalAgency:
    def __init__(self, max_cars=20, max_move=5, gamma=0.9, car_revenue=10, move_cost=2):
        self.MAX_CARS = max_cars
        self.MAX_MOVE = max_move
        self.GAMMA = gamma
        self.CAR_REVENUE = car_revenue
        self.MOVE_COST = move_cost

        self.value = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        self.policy = np.zeros(self.value.shape)

    def expected_return(self, state, action):
        returns = -self.MOVE_COST * abs(action)

        location_1_cars = int(min(state[0] - action, self.MAX_CARS))
        location_2_cars = int(min(state[1] + action, self.MAX_CARS))

        # go through all possible rental requests
        for rental_request_first_loc in range(POISSON_UPPER_BOUND):
            for rental_request_second_loc in range(POISSON_UPPER_BOUND):
                # probability for current combination of rental requests
                prob = poisson_probability(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * \
                    poisson_probability(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

                num_of_cars_first_loc = location_1_cars
                num_of_cars_second_loc = location_2_cars

                # valid rental requests should be less than actual # of cars
                valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
                valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

                # get credits for renting
                reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT
                num_of_cars_first_loc -= valid_rental_first_loc
                num_of_cars_second_loc -= valid_rental_second_loc

                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(
                            returned_cars_first_loc, RETURNS_FIRST_LOC) * poisson_probability(returned_cars_second_loc, RETURNS_SECOND_LOC)
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        prob_ = prob_return * prob
                        returns += prob_ * (reward + self.GAMMA *
                                            self.value[num_of_cars_first_loc_, num_of_cars_second_loc_])
        return returns

    def evaluate_policy(self, epsilon=1e-4):
        max_value_change = 100
        while max_value_change > epsilon:
            prev_state_value = self.value.copy()
            for i in range(self.MAX_CARS + 1):
                for j in range(self.MAX_CARS + 1):
                    self.value[i, j] = self.expected_return([i, j], self.policy[i, j])
            max_value_change = abs(prev_state_value - self.value).max()
            print('max value change {}'.format(max_value_change))

    def evaluate(self):
        iterations = 0
        while iterations < 10:
            print('Iteration', iterations)
            iterations += 1
            self.evaluate_policy()

        return


def main():
    jacks_agency = RentalAgency()
    jacks_agency.evaluate()


main()
