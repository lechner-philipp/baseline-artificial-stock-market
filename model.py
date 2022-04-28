import collections
import pathlib
import random
from datetime import datetime

import jsonpickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, interpolate
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
import random

from numpy.random import default_rng

from order import OrderType, Order
from settings import Settings


class Market:
    def __init__(self, _settings: Settings):

        self.settings = _settings
        self.periods = _settings.periods
        self.risk_free_rate = _settings.risk_free_rate
        self.share_price = []
        self.dividend = []

        self.agents = []
        self.orders = []
        self.volume = []

        self.results = {}

    def __str__(self):
        return "Market()"

    def initialize_market(self):
        self.share_price = [self.settings.initial_share_price]
        self.dividend = [self.settings.dividend]

        self.agents = [Agent(self, i, self.settings.initial_cash, self.settings.initial_shares, self.settings.std) for i
                       in
                       range(self.settings.number_agents)]

    def update(self):
        self.dividend.append(self.dividend[0])

    def calculate_results(self):

        daily_returns = []
        for time in range(1, len(self.share_price)):
            daily_return = (self.share_price[time] - self.share_price[time - 1]) / self.share_price[time - 1]
            daily_returns.append(daily_return)

        mu, std = norm.fit(daily_returns)

        # Calculate the auto-correlation
        squared_daily_returns = []
        for i in daily_returns:
            squared_daily_returns.append(i ** 2)

        absolute_daily_returns = []
        for i in daily_returns:
            absolute_daily_returns.append(abs(i))

        raw_acf = list(acf(np.array(daily_returns), nlags=200))
        squared_acf = list(acf(np.array(squared_daily_returns), nlags=200))
        absolute_acf = list(acf(np.array(absolute_daily_returns), nlags=200))

        for i in [raw_acf, squared_acf, absolute_acf]:
            i.pop(0)

        self.results["daily_returns"] = daily_returns
        self.results["mu"] = mu
        self.results["std"] = std
        self.results["kurtosis"] = stats.kurtosis(daily_returns)
        self.results["skew"] = stats.skew(np.array(daily_returns))
        self.results["raw_acf"] = raw_acf
        self.results["squared_acf"] = squared_acf
        self.results["absolute_acf"] = absolute_acf

        return self.results

    def visualize(self):
        daily_returns = self.results["daily_returns"]
        raw_acf = self.results["raw_acf"]
        squared_acf = self.results["squared_acf"]
        absolute_acf = self.results["absolute_acf"]
        mu = self.results["mu"]
        std = self.results["std"]
        _kurtosis = self.results["kurtosis"]
        _skew = self.results["skew"]

        plt.rcParams["figure.figsize"] = (8, 9)
        fig, axs = plt.subplots(3)

        # Price Time Series
        axs[0].plot(self.share_price)

        # Return Distribution
        plot_return_distribution(axs[1], daily_returns, mu, std, _skew, _kurtosis)

        # Auto Correlation
        axs[2].plot(raw_acf, label="raw")
        axs[2].plot(squared_acf, label="squared")
        axs[2].plot(absolute_acf, label="absolute")

        fig.tight_layout()
        plt.legend()
        plt.show()

        # plt.savefig("share_price_random_walk", dpi=500)

    def clear_market(self, time):
        buy_orders = [i for i in self.orders if i.type == OrderType.buy]
        sell_orders = [i for i in self.orders if i.type == OrderType.sell]

        if len(buy_orders) != 0 and len(sell_orders) != 0:
            buy_order_frequency = collections.Counter(buy_orders)
            buy_order_frequency = collections.OrderedDict(reversed(sorted(buy_order_frequency.items())))

            sell_order_frequency = collections.Counter(sell_orders)
            sell_order_frequency = collections.OrderedDict(sorted(sell_order_frequency.items()))

            supply = {}
            cumulative_supply = 0
            for i in sell_order_frequency:
                cumulative_supply = sell_order_frequency[i] + cumulative_supply
                supply[i.value] = cumulative_supply

            demand = {}
            cumulative_demand = 0
            for i in buy_order_frequency:
                cumulative_demand = buy_order_frequency[i] + cumulative_demand
                demand[i.value] = cumulative_demand

            # plt.plot(demand.keys(), demand.values(), label="Demand")
            # plt.plot(supply.keys(), supply.values(), label="Supply")
            # plt.legend()

            clearing_price = None
            try:
                interpolated_supply = interpolate.interp1d(list(supply.keys()), list(supply.values()))
            except ValueError:
                clearing_price = self.share_price[time - 1]
                pass
            try:
                interpolated_demand = interpolate.interp1d(list(demand.keys()), list(demand.values()))
            except ValueError:
                clearing_price = self.share_price[time - 1]
                pass

            if clearing_price == None:
                difference = {}
                for i in list(np.arange(self.share_price[time - 1] - self.settings.std * 2,
                                        self.share_price[time - 1] + self.settings.std * 2, self.settings.std/10)):
                    try:
                        difference[i] = abs(interpolated_supply(i) - interpolated_demand(i))
                    except ValueError:
                        pass

                try:
                    clearing_price = list(difference.keys())[list(difference.values()).index(min(difference.values()))]
                except ValueError:
                    clearing_price = self.share_price[time - 1]
                    print("No clearing price found")

            valid_buy_orders = [order for order in buy_orders if order.value >= clearing_price]
            valid_sell_orders = [order for order in sell_orders if order.value <= clearing_price]

            difference_number_orders = len(valid_buy_orders) - len(valid_sell_orders)

            if difference_number_orders > 0:
                for i in range(difference_number_orders):
                    # Most efficient way to remove a random element from a list
                    i = random.randrange(len(valid_buy_orders))
                    valid_buy_orders[i], valid_buy_orders[-1] = valid_buy_orders[-1], valid_buy_orders[i]
                    valid_buy_orders.pop()

            if difference_number_orders < 0:
                for i in range(difference_number_orders * -1):
                    # Most efficient way to remove a random element from a list
                    i = random.randrange(len(valid_sell_orders))
                    valid_sell_orders[i], valid_sell_orders[-1] = valid_sell_orders[-1], valid_sell_orders[i]
                    valid_sell_orders.pop()

            for order in valid_buy_orders:
                order.execute(time, clearing_price)

            for order in valid_sell_orders:
                order.execute(time, clearing_price)

            self.share_price.append(clearing_price)
            # plt.show()

        elif sell_orders == 0:
            self.share_price.append(max(buy_orders))
        elif buy_orders == 0:
            self.share_price.append(min(sell_orders))

    def run(self, visualization=False):
        self.initialize_market()
        for time in range(1, self.periods + 1):
            self.orders = []
            for agent in self.agents:
                agent.order(time)

            self.clear_market(time)
            self.update()

            for agent in self.agents:
                agent.update(time)

        self.calculate_results()
        if visualization:
            self.visualize()

        return self.results

    def export_data(self, _name):
        with open(str(pathlib.Path(__file__).parent.resolve()) + "\\exports\\" + _name + "_" + datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S") + ".json", "w") as f:
            f.write(jsonpickle.encode(self.__dict__))

    def import_data(self, path):
        with open(path, "r") as f:
            self.__dict__ = jsonpickle.decode(f.read())


def plot_return_distribution(ax, _returns, _mu, _std, _skew, _kurtosis, title=True):

    # Return Distribution
    ax.hist(_returns, bins=500, density=True)
    x = np.linspace(-0.2, 0.2, 500)
    p = norm.pdf(x, _mu, _std)

    ax.plot(x, p, 'k', linewidth=2)

    if title:
        ax.set_title(
            r"$\mu$={:.4f}, $\sigma$={:.4f}, skewness={:.4f}, kurtosis={:.4f}".format(_mu, _std, _skew, _kurtosis))

    ax.set_xlim([-0.05, 0.05])
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Probability")

    print(_mu, _std, _skew, _kurtosis)


# ----------------------------------------------------------------------------------------------------------------------
# AGENT
# ----------------------------------------------------------------------------------------------------------------------


class Agent:
    def __init__(self, market: Market, index, cash, number_shares, std):
        self.market = market
        self.index = index
        self.cash = [float(cash)]
        self.number_shares = [int(number_shares)]
        self.wealth = [float(self.cash[0] + self.number_shares[0] * self.market.share_price[0])]
        self.equity = [float(self.wealth[0] - self.cash[0])]
        self.std = std

    def update(self, time):
        self.number_shares.append(self.number_shares[0])

        # Variate of Equation 13
        if len(self.cash) == time:
            self.cash.append(self.cash[time - 1])

        self.cash[time] += self.market.dividend[time] * self.number_shares[time]
        self.cash[time] = (1 + self.market.risk_free_rate) * self.cash[time]
        self.wealth.append(self.cash[time] + self.market.share_price[time] * self.number_shares[time])
        self.equity.append(self.wealth[time] - self.cash[time])
        return self.cash, self.wealth

    def order(self, time):
        rng = default_rng()
        mean, standard_deviation = self.market.share_price[time - 1], self.std

        order_type = random.choice([OrderType.buy, OrderType.sell])
        order_value = rng.normal(mean, standard_deviation, 1)[0]

        # Check if agent meets conditions to execute the order
        if order_value < 0:
            return

        if order_type == OrderType.buy:
            if order_value > self.cash[time - 1]:
                return
        else:
            if self.number_shares == 0:
                return

        order = Order(self, order_type, order_value)
        self.market.orders.append(order)

    def __str__(self):
        return str(str(self.__dict__))
