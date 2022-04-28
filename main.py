import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
import numpy as np
from numpy.polynomial import polynomial
import json

from model import Market, plot_return_distribution
from settings import Settings
from analyze import multi_analyze_results, analyze_results


def main():
    start_time = datetime.now()

    optimize(50, 2, 3, 0.05)

    print("-" * 30)
    print(f"FINISHED in {datetime.now() - start_time}")


def optimize(runs, start, end, step):
    for i in list(np.arange(start, end, step)):
        i = float(i)
        _settings = Settings(periods=23673, number_agents=50, initial_share_price=100, risk_free_rate=0.015,
                             dividend=5, initial_cash=1000, initial_shares=1, std=i)
        multi_run("{:.2f}".format(i), runs, _settings)


def spx():
    with open("spx/spx_results.json", "r") as f:
        results = json.load(f)

    mu = results["mu"]
    std = results["std"]
    skew = results["skew"]
    kurtosis = results["kurtosis"]
    daily_returns = results["daily_returns"]
    raw_acf = results["raw_acf"]
    squared_acf = results["squared_acf"]
    absolute_acf = results["absolute_acf"]

    # 0.0003061881281387075 0.011964309127722885 -0.12028614737963762 17.517850200685995
    print(mu, std, skew, kurtosis)

    fig, ax = plt.subplots(1, 1)

    raw_acf_coefficients = polynomial.polyfit(range(1, len(raw_acf) + 1), raw_acf, 9)
    raw_acf_model = polynomial.Polynomial(raw_acf_coefficients)

    squared_acf_coefficients = polynomial.polyfit(range(1, len(squared_acf) + 1), squared_acf, 9)
    squared_acf_model = polynomial.Polynomial(squared_acf_coefficients)

    absolute_acf_coefficients = polynomial.polyfit(range(1, len(absolute_acf) + 1), absolute_acf, 9)
    absolute_acf_model = polynomial.Polynomial(absolute_acf_coefficients)

    plt.plot(raw_acf, label="raw")
    plt.plot(raw_acf_model(range(1, len(raw_acf) + 1)), label="raw model", color="grey")

    plt.plot(squared_acf, label="squared")
    plt.plot(squared_acf_model(range(1, len(squared_acf) + 1)), label="squared model", color="grey")

    plt.plot(absolute_acf, label="absolute")
    plt.plot(absolute_acf_model(range(1, len(absolute_acf) + 1)), label="absolute model", color="grey")

    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.legend()
    plt.show()

    # plot_return_distribution(ax, daily_returns, mu, std, skew, kurtosis, title=False)
    # plt.savefig("images/spx_autocorrelation.png", dpi=600)


def multi_run(_name, _number_runs, _settings):
    results = {
        "settings": _settings.__dict__,
        "results": []
    }

    for i in range(1, _number_runs + 1):
        start_time = datetime.now()

        market = Market(_settings)
        result = market.run()
        results["results"].append(result.copy())

        print("-" * 30)
        print(f"FINISHED Run {i} in {datetime.now() - start_time}")
        print("-" * 30)

    with open(f"results/results_{_name}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
