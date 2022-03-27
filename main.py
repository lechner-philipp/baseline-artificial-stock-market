import matplotlib.pyplot as plt
from numpy.random import default_rng
import numpy as np
import collections
from scipy import interpolate

import random

random.choice([-1, 1])


def main():
    market = Market()
    market.run()


class Market:
    def __init__(self):
        self.share_price = []
        self.risk_free_rate = 0.015
        self.dividend = []

        self.agents = []
        self.orders = []

    def __str__(self):
        return "Market()"

    def initialize_market(self):
        self.share_price = [100]
        self.dividend = [5]

        self.agents = [Agent(self, i, 200, 1) for i in range(100)]

    def update(self, time):
        self.share_price.append(self.share_price[0])
        self.dividend.append(self.dividend[0])

    def final(self):
        plt.plot(self.agents[0].wealth)
        plt.show()

    def market_clearing(self):
        buy_orders = [i for i in self.orders if i > 0]
        sell_orders = [i for i in self.orders if i < 0]

        # Make sell order sign positive
        for i in sell_orders:
            sell_orders[sell_orders.index(i)] = i * -1

        buy_order_frequency = collections.Counter(buy_orders)
        buy_order_frequency = collections.OrderedDict(reversed(sorted(buy_order_frequency.items())))

        sell_order_frequency = collections.Counter(sell_orders)
        sell_order_frequency = collections.OrderedDict(sorted(sell_order_frequency.items()))

        supply = {}
        cumulative_supply = 0
        for i in sell_order_frequency:
            cumulative_supply = sell_order_frequency[i] + cumulative_supply
            supply[i] = cumulative_supply

        demand = {}
        cumulative_demand = 0
        for i in buy_order_frequency:
            cumulative_demand = buy_order_frequency[i] + cumulative_demand
            demand[i] = cumulative_demand
        plt.plot(demand.keys(), demand.values(), label="Demand")
        plt.plot(supply.keys(), supply.values(), label="Supply")
        plt.legend()

        interpolated_supply = interpolate.interp1d(list(supply.keys()), list(supply.values()))
        interpolated_demand = interpolate.interp1d(list(demand.keys()), list(demand.values()))

        # TODO: Optimize: Directed Walk with expected Value
        difference = {}
        for i in list(np.arange(90, 110, 0.1)):
            try:
                difference[i] = abs(interpolated_supply(i) - interpolated_demand(i))
            except ValueError:
                pass
        print(list(difference.keys())[list(difference.values()).index(min(difference.values()))])
        plt.show()

    def run(self):
        self.initialize_market()
        for time in range(1):
            self.orders = []
            for agent in self.agents:
                agent.order(time)

            self.market_clearing()
            self.update(time)

            for agent in self.agents:
                agent.update(time)

        # self.final()


class Agent:
    def __init__(self, market: Market, index, cash, number_shares):
        self.market = market
        self.index = index
        self.cash = [float(cash)]
        self.number_shares = [int(number_shares)]
        self.wealth = [float(self.cash[0] + self.number_shares[0] * self.market.share_price[0])]
        self.equity = [float(self.wealth[0] - self.cash[0])]

    def update(self, time):
        self.number_shares.append(self.number_shares[0])

        # Variate of Equation 13
        self.cash.append((1 + self.market.risk_free_rate) * self.cash[time - 1])
        self.wealth.append(self.cash[time] + self.market.share_price[time] * self.number_shares[time])
        self.equity.append(self.wealth[time] - self.cash[time])
        return self.cash, self.wealth

    def order(self, time):
        rng = default_rng()
        mean, standard_deviation = 100, 10

        # 1 = Buy; -1 = Sell
        order_type = random.choice([-1, 1])
        order = round(rng.normal(mean, standard_deviation, 1)[0], 0) * order_type

        # Check if agent meets conditions to execute the order
        if order_type == 1:
            if order > self.cash[time - 1]:
                return
        else:
            if self.number_shares == 0:
                return

        self.market.orders.append(order)

    def __str__(self):
        return str(str(self.__dict__))


if __name__ == "__main__":
    main()
