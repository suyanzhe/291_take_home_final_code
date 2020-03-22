import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class epsilon_function():
    def __init__(self, string, function):
        self.string = string
        self.function = function

    def compute(self, t):
        return self.function(t)

    def __str__(self):
        return self.string

class coin():
    def __init__(self, avg):
        if avg > 1 or avg < 0:
            raise ValueError("Average reward must be between 0 and 1")
        self.avg = avg
        self.mu = -1
        self.count = 0
    
    def roll_coin(self):
        reward = np.random.random() < self.avg
        self.mu = (self.mu * self.count + reward) / (self.count + 1)
        self.count += 1
        return reward

    def compute_ucb(self, t):
        return self.mu + np.sqrt(2 * np.log(t) / self.count)

    def __eq__(self, coin2):
        return self.mu == coin2.mu if type(coin2) == type(self) else NotImplemented

    def __lt__(self, coin2):
        return self.mu < coin2.mu if type(coin2) == type(self) else NotImplemented
    
    def __str__(self):
        return str(self.avg)

class epsilon_greedy():
    def __init__(self, coin1_avg, coin2_avg, epsilon = None):
        if coin1_avg == coin2_avg:
            raise ValueError("The coins must have different average reward")
        self.coin1 = coin(coin1_avg)
        self.coin2 = coin(coin2_avg)
        self.delta = np.abs(self.coin1.avg - self.coin2.avg)
        self.good_coin = self.coin1 if coin1_avg > coin2_avg else self.coin2
        self.is_function_epsilon = False
        if epsilon is not None:
            try:
                self.epsilon = float(epsilon)
            except (ValueError, TypeError):
                if str(type(epsilon)) == "<class '__main__.epsilon_function'>":
                    self.epsilon = epsilon
                    self.is_function_epsilon = True
        else:
            self.epsilon = 0.01
        self.regret = 0
        self.regrets = []
        self.reward = 0
        self.rewards = []
    
    def run(self, iters):
        self.regret = 0
        self.regrets = []
        self.reward = 0
        self.rewards = []
        for i in range(iters):
            if not self.coin1.count:
                self.reward += self.coin1.roll_coin()
                self.regret += (self.coin1 != self.good_coin) * self.delta
            elif not self.coin2.count:
                self.reward += self.coin2.roll_coin()
                self.regret += (self.coin2 != self.good_coin) * self.delta
            else:
                coin = None
                if np.random.random() < (self.epsilon if not self.is_function_epsilon else self.epsilon.compute(i)):
                    coin = min(self.coin1, self.coin2)
                else:
                    coin = max(self.coin1, self.coin2)
                self.reward += coin.roll_coin()
                self.regret += (coin != self.good_coin) * self.delta
            self.rewards.append(self.reward)
            self.regrets.append(self.regret)
        return (r"$\epsilon$-greedy with $\epsilon$ = " + str(self.epsilon), self.rewards, self.regrets)

class ucb():
    def __init__(self, coin1_avg, coin2_avg):
        if coin1_avg == coin2_avg:
            raise ValueError("The coins must have different average reward")
        self.coin1 = coin(coin1_avg)
        self.coin2 = coin(coin2_avg)
        self.delta = np.abs(self.coin1.avg - self.coin2.avg)
        self.good_coin = self.coin1 if coin1_avg > coin2_avg else self.coin2
        self.regret = 0
        self.regrets = []
        self.reward = 0
        self.rewards = []

    def run(self, iters):
        self.regret = 0
        self.regrets = []
        self.reward = 0
        self.rewards = []
        for i in range(iters):
            if not self.coin1.count:
                self.reward += self.coin1.roll_coin()
                self.regret += (self.coin1 != self.good_coin) * self.delta
            elif not self.coin2.count:
                self.reward += self.coin2.roll_coin()
                self.regret += (self.coin2 != self.good_coin) * self.delta
            else:
                ucb1 = self.coin1.compute_ucb(i)
                ucb2 = self.coin2.compute_ucb(i)
                coin = self.coin1 if ucb1 > ucb2 else self.coin2
                self.reward += coin.roll_coin()
                self.regret += (coin != self.good_coin) * self.delta
            self.regrets.append(self.regret)
            self.rewards.append(self.reward)
        return ("Upper-confidence bound", self.rewards, self.regrets)

def draw_curves(data_group, is_regrets = False, fig_name = None, colors = None):
    plt.rcParams["font.family"] = "STIX"
    plt.rcParams["mathtext.fontset"] = "stix"
    figure = plt.figure(figsize=(5, 3))
    assert len(colors) == len(data_group)
    data_index = is_regrets + 1
    if colors is not None:
        for data, c in zip(data_group, colors):
            plt.plot(range(len(data[data_index])), data[data_index], label = data[0], color = c)
    else:
        for data in data_group:
            plt.plot(range(len(data[data_index])), data[data_index], label = data[0])
    plt.legend()
    plt.xlabel("Iteration # (time)")
    plt.ylabel("Accumulated regret" if is_regrets else "Accumulated reward")
    plt.subplots_adjust(0.15, 0.175, 0.975, 0.975)
    if fig_name is not None:
        plt.savefig(fig_name + ("regret" if is_regrets else "reward") + ".pdf", dpi = 3584)
    plt.show()

def optimal_epsilon(t):
    return np.power(t, -1 / 3) * np.power(np.log(t), 1 / 3) / 10

def main():
    c = ["#F44336", "#4CAF50", "#2196F3", "#000000"]
    for coins in [[0.1, 0.8], [0.4, 0.6], [0.44, 0.47]]:
        data_group = [experiment.run(1 << 20) for experiment in [epsilon_greedy(coins[0], coins[1], 0.01), epsilon_greedy(coins[0], coins[1], 0.1), epsilon_greedy(coins[0], coins[1], epsilon_function(r"$\frac{t^{-\frac{1}{3}}\log t^{\frac{1}{3}}}{10}$", optimal_epsilon)), ucb(coins[0], coins[1])]]
        for ir in [False, True]:
            draw_curves(data_group, is_regrets = ir, fig_name = str(coins[0]) + "-" + str(coins[1]), colors = c[:len(data_group)])

if __name__ == "__main__":
    main()