import json

import jsonpickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
from numpy.polynomial import polynomial
from model import Market
from settings import Settings


def multi_analyze_results(start, end, step, cached=True):
    if not cached:
        _settings = Settings(periods=1, number_agents=25, initial_share_price=100, risk_free_rate=0.015, dividend=5,
                             initial_cash=1000, initial_shares=1, std=1.75)

        results = []
        for i in list(np.arange(start, end, step)):
            results.append(analyze_results("results/results_{:.2f}.json".format(i), _settings, ))

        with open("analyzed_results.json", "w") as f:
            json.dump(results, f)
    else:
        with open("analyzed_results.json", "r") as f:
            results = json.load(f)

    kurtosis_dict = {}
    std_dict = {}
    raw_acf_dict = {}
    squared_acf_coefficients = {}
    absolute_acf_dict = {}
    skewness_dict = {}
    for i in range(1, len(results)):
        kurtosis_dict[results[i]["order_std"]] = results[i]["kurtosis"]
        std_dict[results[i]["order_std"]] = results[i]["std"]

        raw_acf_dict[results[i]["order_std"]] = results[i]["raw_acf_accuracy"]
        squared_acf_coefficients[results[i]["order_std"]] = results[i]["squared_acf_accuracy"]
        absolute_acf_dict[results[i]["order_std"]] = results[i]["absolute_acf_accuracy"]

        skewness_dict[results[i]["order_std"]] = results[i]["skew"]

    plt.rcParams["figure.figsize"] = (6.4, 4.8 * 4)
    fig, axs = plt.subplots(4)

    # --------------------------------------------------
    # Plot STD
    # --------------------------------------------------
    x = list(std_dict.keys())
    y = list(std_dict.values())
    axs[0].plot(x, y)

    @np.vectorize
    def spx_standard_deviation(x):
        return 0.011964309127722885

    x = np.array(x)
    axs[0].plot(x, spx_standard_deviation(x), label="S&P 500 Daily Return Standard Deviation")
    k, d, r, p, std = stats.linregress(x, y)

    def linear_function(x):
        return k * x + d

    model = list(map(linear_function, x))
    axs[0].plot(x, model)

    axs[0].set_title(r"Daily Return Standard Deviation: $\sigma_o$={:.2f}".format((0.011964309127722885 - d) / k))
    axs[0].set_xlabel(r"$\sigma_o$")
    axs[0].set_ylabel("Daily Return Standard Deviation")
    axs[0].legend()

    # --------------------------------------------------
    # Plot Kurtosis
    # --------------------------------------------------
    x = list(kurtosis_dict.keys())
    y = list(kurtosis_dict.values())
    axs[1].plot(x, y)

    @np.vectorize
    def spx_kurtosis(x):
        return 17.517850200685995

    axs[1].plot(x, spx_kurtosis(x), label="S&P 500 Daily Return Kurtosis")
    axs[1].legend()

    k, d, r, p, std = stats.linregress(x, y)

    def linear_function(x):
        return k * x + d

    axs[1].plot(x, list(map(linear_function, x)))
    axs[1].set_title(r"Kurtosis: $\sigma_o$={:.2f}".format((17.517850200685995 - d) / k))
    axs[1].set_xlabel(r"$\sigma_o$")
    axs[1].set_ylabel("Kurtosis")

    # --------------------------------------------------
    # Plot Skewness
    # --------------------------------------------------
    x = list(skewness_dict.keys())
    y = list(skewness_dict.values())
    axs[2].plot(x, y)

    axs[2].set_title(r"Skewness")
    axs[2].set_xlabel(r"$\sigma_o$")
    axs[2].set_ylabel("Skewness")

    # --------------------------------------------------
    # Plot Auto Correlation
    # --------------------------------------------------

    axs[3].set_title("Auto Correlation Error")

    x = list(squared_acf_coefficients.keys())
    y = list(squared_acf_coefficients.values())
    axs[3].plot(x, y, label="Squared Autocorrelation Error")

    squared_acf_coefficients = polynomial.polyfit(x, y, 6)
    squared_acf_model = polynomial.Polynomial(squared_acf_coefficients)
    axs[3].plot(x, list(map(squared_acf_model, x)), color="grey")
    squared_acf_min = minimize(squared_acf_model, x0=np.array(2)).x
    print(f"{squared_acf_min=}")
    axs[3].plot(squared_acf_min, squared_acf_model(squared_acf_min), color="red", marker="o")

    x = list(absolute_acf_dict.keys())
    y = list(absolute_acf_dict.values())
    axs[3].plot(x, y, label="Absolute Autocorrelation Error")

    absolute_acf_coefficients = polynomial.polyfit(x, y, 6)
    absolute_acf_model = polynomial.Polynomial(absolute_acf_coefficients)
    axs[3].plot(x, list(map(absolute_acf_model, x)), color="grey")
    absolute_acf_min = minimize(absolute_acf_model, x0=np.array(2)).x
    print(f"{absolute_acf_min=}")
    axs[3].plot(absolute_acf_min, absolute_acf_model(absolute_acf_min), color="red", marker="o")

    def total_error_model(x):
        return squared_acf_model(x) + absolute_acf_model(x)

    axs[3].plot(x, list(map(total_error_model, x)), label="Total Autocorreclation Error")
    total_error_min = minimize(total_error_model, x0=np.array(2)).x
    print(f"{total_error_min=}")
    axs[3].plot(total_error_min, total_error_model(total_error_min), color="red", marker="o")

    axs[3].set_xlabel(r"$\sigma_o$")
    axs[3].set_ylabel("Auto Correlation Error")

    plt.legend()
    fig.tight_layout()
    plt.show()


def analyze_results(path, _settings, visualize=False, plot_autocorrelation=False):
    market = Market(_settings)

    with open(path, "r") as f:
        results = jsonpickle.decode(f.read())

    order_std = results["settings"]["std"]

    with open("spx/spx_results.json", "r") as f:
        spx = json.load(f)

    mu_list = []
    std_list = []
    kurtosis_list = []
    skew_list = []
    daily_returns = []
    for i in results["results"]:
        daily_returns += i["daily_returns"]
        mu_list.append(i["mu"])
        std_list.append(i["std"])
        kurtosis_list.append(i["kurtosis"])
        skew_list.append(i["skew"])

    market.results["daily_returns"] = daily_returns
    market.results["mu"] = np.average(mu_list)
    market.results["std"] = np.average(std_list)
    market.results["kurtosis"] = np.average(kurtosis_list)
    market.results["skew"] = np.average(skew_list)

    for acf_type in ["raw_acf", "squared_acf", "absolute_acf"]:
        market.results[acf_type] = []
        for i in range(len(results["results"][0][acf_type])):

            total = 0
            for j in results["results"]:
                total += j[acf_type][i]

            market.results[acf_type].append(total / len(results["results"]))

    market.results["raw_acf_accuracy"] = sum(np.absolute(np.subtract(market.results["raw_acf"], spx["raw_acf"])))
    market.results["squared_acf_accuracy"] = sum(
        np.absolute(np.subtract(market.results["squared_acf"], spx["squared_acf"])))
    market.results["absolute_acf_accuracy"] = sum(
        np.absolute(np.subtract(market.results["absolute_acf"], spx["absolute_acf"])))

    if visualize:
        market.visualize()

    if plot_autocorrelation:
        # plt.plot(market.results["raw_acf"], label="raw")
        # plt.plot(spx["raw_acf"], label="S&P 500 raw")

        plt.plot(spx["absolute_acf"], label="S&P 500 Absolute Autocorrelation", color="green")
        plt.plot(market.results["absolute_acf"], label="AL-ASM Absolute Autocorrelation", color="blue")

        plt.plot(spx["squared_acf"], label="S&P 500 Squared Autocorrelation", color="orange")
        plt.plot(market.results["squared_acf"], label="BL-ASM Squared Autocorrelation", color="red")

        plt.legend()
        plt.show()

    results = market.results
    results["order_std"] = order_std

    return results
