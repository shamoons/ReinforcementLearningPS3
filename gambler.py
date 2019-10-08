import random


class Gambler:
    def __init__(self, capital=0, prob_h=0.4):
        self.capital = capital
        self.reward = 0
        self.prob_h = prob_h

    def play(self, action):
        win = True if random.random() < self.prob_h else False

        if win:
            self.capital = self.capital + action
        else:
            self.capital = self.capital - action


gambler = Gambler()

for i in range(0, 100):
    gambler.play()
