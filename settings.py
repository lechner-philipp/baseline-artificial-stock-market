class Settings:
    def __init__(self, periods, number_agents, initial_share_price, risk_free_rate, dividend, initial_cash,
                 initial_shares, std):
        # Market Settings
        self.periods = periods
        self.number_agents = number_agents
        self.initial_share_price = initial_share_price
        self.risk_free_rate = risk_free_rate
        self.dividend = dividend

        # Agent Settings
        self.initial_cash = initial_cash
        self.initial_shares = initial_shares
        self.std = std