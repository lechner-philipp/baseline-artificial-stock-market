from enum import Enum


class OrderType(Enum):
    buy = 1
    sell = -1


class OrderStatus(Enum):
    pending = 0
    fulfilled = 1
    rejected = -1


class Order:
    def __init__(self, agent, order_type, order_value):
        self.agent = agent
        self.type = order_type
        self.value = order_value
        # self.status = OrderStatus.pending

    def __str__(self):
        return "Order: " + str(self.__dict__)

    def execute(self, time, price):
        if self.type == OrderType.buy:
            self.agent.cash.append(self.agent.cash[time - 1] - price)
            self.agent.number_shares.append(self.agent.number_shares[time - 1] + 1)
        elif self.type == OrderType.sell:
            self.agent.cash.append(self.agent.cash[time - 1] + price)
            self.agent.number_shares.append(self.agent.number_shares[time - 1] - 1)

    def __lt__(self, other):
        return self.value < other.value