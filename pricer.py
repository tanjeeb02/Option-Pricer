import math as m
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc


class Pricer:

    def __init__(self, S, r, q, T, K, seed=100):
        self.S = S  # Spot
        self.r = r  # Yield
        self.q = q  # Repo Rate
        self.T = T  # Tenor
        self.K = K  # Strike
        self.seed = seed  # Seed

    def set_seed(self):
        np.random.seed(self.seed)

    def d1_european(self, sigma):
        return (m.log(self.S / self.K) + (self.r - self.q + 0.5 * sigma ** 2) * self.T) / (sigma * m.sqrt(self.T))

    def d2_european(self, sigma):
        return self.d1_european(sigma) - sigma * m.sqrt(self.T)

    def european_option(self, sigma, option_type):

        d1 = self.d1_european(sigma)
        d2 = self.d2_european(sigma)

        if option_type == 'call':
            nd1 = norm.cdf(d1)
            nd2 = norm.cdf(d2)
            price = self.S * m.exp(-self.q * self.T) * nd1 - self.K * m.exp(-self.r * self.T) * nd2
        elif option_type == 'put':
            nd1 = norm.cdf(-d1)
            nd2 = norm.cdf(-d2)
            price = self.K * m.exp(-self.r * self.T) * nd2 - self.S * m.exp(-self.q * self.T) * nd1
        else:
            raise ValueError("Please input 'call' or 'put'.")

        return price

    def iv(self, option_premium, option_type):

        sigma_hat = m.sqrt(2 * abs((m.log(self.S / self.K) + (self.r - self.q) * self.T) / self.T))
        tol = 1e-8
        n_max = 100
        sigma_diff = 1
        n = 1
        sigma = sigma_hat

        while sigma_diff >= tol and n < n_max:
            d1_next = self.d1_european(sigma)
            option = self.european_option(sigma, option_type)
            vega = self.S * m.exp(-self.q * self.T) * m.sqrt(self.T) * norm.pdf(d1_next)
            increment = (option - option_premium) / vega
            sigma = sigma - increment
            n += 1
            sigma_diff = abs(increment)

        return sigma

    @staticmethod
    def sigma_asian(sigma, n):
        return sigma * m.sqrt(((n + 1) * (2 * n + 1)) / (6 * n ** 2))

    def miu_asian(self, sigma, n):
        sigma_adj = Pricer.sigma_asian(sigma, n)
        return (self.r - 0.5 * sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_adj ** 2

    def d1_asian(self, sigma, n):
        sigma_adj = Pricer.sigma_asian(sigma, n)
        miu_adj = self.miu_asian(sigma, n)
        return (m.log(self.S / self.K) + (miu_adj + 0.5 * sigma_adj ** 2) * self.T) / (sigma_adj * m.sqrt(self.T))

    def d2_asian(self, d1, sigma, n):
        sigma_adj = Pricer.sigma_asian(sigma, n)
        return d1 - sigma_adj * m.sqrt(self.T)

    def asian_geometric_option(self, sigma, n, option_type, is_monte=False):

        self.set_seed()

        paths = 100000

        # sigma_adj = Pricer.sigma_asian(sigma, n)
        miu_adj = self.miu_asian(sigma, n)

        d1 = self.d1_asian(sigma, n)
        d2 = self.d2_asian(d1, sigma, n)

        nd1_asian_call = norm.cdf(d1)
        nd2_asian_call = norm.cdf(d2)
        nd1_asian_put = norm.cdf(-d1)
        nd2_asian_put = norm.cdf(-d2)

        asian_call_price_closed = m.exp(-self.r * self.T) * (
                self.S * m.exp(miu_adj * self.T) * nd1_asian_call - self.K * nd2_asian_call)
        asian_put_price_closed = m.exp(-self.r * self.T) * (
                self.K * nd2_asian_put - self.S * m.exp(miu_adj * self.T) * nd1_asian_put)

        if is_monte:

            Z = np.random.normal(0, 1, (paths, n))  # Generate (Mxn) random variables
            dt = self.T / n
            drift = (self.r - 0.5 * sigma ** 2) * dt
            log_price_paths = np.log(self.S) + np.cumsum((drift + sigma * np.sqrt(dt) * Z), axis=1)

            geo_means = np.exp(np.mean(log_price_paths, axis=1))
            geo_call_payoff = np.maximum(geo_means - self.K, 0) * np.exp(-self.r * self.T)
            geo_put_payoff = np.maximum(self.K - geo_means, 0) * np.exp(-self.r * self.T)
            geo_call_price = np.mean(geo_call_payoff)
            geo_put_price = np.mean(geo_put_payoff)

            if option_type == 'call':
                return geo_call_price
            elif option_type == 'put':
                return geo_put_price
            else:
                return ValueError("Please specify an option type.")

        elif not is_monte:
            if option_type == 'call':
                return asian_call_price_closed
            elif option_type == 'put':
                return asian_put_price_closed
            else:
                return ValueError("Please specify an option type.")

    def asian_arithmetic_option(self, sigma, n, paths, option_type, with_cv=True):

        self.set_seed()

        Z = np.random.normal(0, 1, (paths, n))  # Generate (Mxn) random variables
        dt = self.T / n
        drift = (self.r - 0.5 * sigma ** 2) * dt
        log_price_paths = np.log(self.S) + np.cumsum((drift + sigma * np.sqrt(dt) * Z), axis=1)

        # Find geometric and arithmetic means with simulated price paths
        geo_means = np.exp(np.mean(log_price_paths, axis=1))
        arith_means = np.mean(np.exp(log_price_paths), axis=1)

        # Find discounted geometric and arithmetic payoffs
        geo_call_payoff = np.maximum(geo_means - self.K, 0) * np.exp(-self.r * self.T)
        geo_put_payoff = np.maximum(self.K - geo_means, 0) * np.exp(-self.r * self.T)
        arith_call_payoff = np.maximum(arith_means - self.K, 0) * np.exp(-self.r * self.T)
        arith_put_payoff = np.maximum(self.K - arith_means, 0) * np.exp(-self.r * self.T)

        # Monte-Carlo output for geometric and arithmetic call/put
        geo_call_price = np.mean(geo_call_payoff)
        geo_put_price = np.mean(geo_put_payoff)
        arith_call_price = np.mean(arith_call_payoff)
        arith_put_price = np.mean(arith_put_payoff)

        stdev_call = np.std(arith_call_payoff)
        stdev_put = np.std(arith_put_payoff)

        if with_cv:

            covXY_call = np.cov(arith_call_payoff, geo_call_payoff)[0, 1]
            theta_call = covXY_call / np.var(geo_call_payoff)
            covXY_put = np.cov(arith_put_payoff, geo_put_payoff)[0, 1]
            theta_put = covXY_put / np.var(geo_put_payoff)

            if option_type == 'call':
                adjusted_arith_call_price = arith_call_price + theta_call * (self.asian_geometric_option(sigma, n, option_type='call', is_monte=False) - geo_call_price)
                confmc = [adjusted_arith_call_price - 1.96 * stdev_call/m.sqrt(paths), adjusted_arith_call_price + 1.96 * stdev_call/m.sqrt(paths)]
                return adjusted_arith_call_price, confmc
            elif option_type == 'put':
                adjusted_arith_put_price = arith_put_price + theta_put * (self.asian_geometric_option(sigma, n, option_type='put', is_monte=False) - geo_put_price)
                confmc = [adjusted_arith_put_price - 1.96 * stdev_put / m.sqrt(paths), adjusted_arith_put_price + 1.96 * stdev_put / m.sqrt(paths)]
                return adjusted_arith_put_price, confmc
            else:
                raise ValueError("Please specify 'call' or 'put'!")

        elif not with_cv:
            if option_type == 'call':
                confmc = [arith_call_price - 1.96 * stdev_call / m.sqrt(paths), arith_call_price + 1.96 * stdev_call / m.sqrt(paths)]
                return arith_call_price, confmc
            elif option_type == 'put':
                confmc = [arith_put_price - 1.96 * stdev_call / m.sqrt(paths), arith_put_price + 1.96 * stdev_call / m.sqrt(paths)]
                return arith_put_price, confmc
            else:
                raise ValueError("Please specify 'call' or 'put'!")

    @staticmethod
    def sigma_basket(sigma_1, sigma_2, rho):
        return m.sqrt((sigma_1 ** 2 + sigma_2 ** 2 + 2 * sigma_1 * sigma_2 * rho)) / 2

    def mu_basket(self, sigma_1, sigma_2, rho):
        sigma_adj = Pricer.sigma_basket(sigma_1, sigma_2, rho)
        return self.r - 0.5 * (sigma_1 ** 2 + sigma_2 ** 2) / 2 + 0.5 * sigma_adj ** 2

    def d1(self, S2, sigma_1, sigma_2, rho):
        sigma_adj = Pricer.sigma_basket(sigma_1, sigma_2, rho)
        S_basket = m.sqrt(self.S * S2)
        return (m.log(S_basket / self.K) + (self.r + 0.5 * sigma_adj ** 2) * self.T) / (sigma_adj * m.sqrt(self.T))

    def d2(self, S2, sigma_1, sigma_2, rho):
        sigma_adj = Pricer.sigma_basket(sigma_1, sigma_2, rho)
        return self.d1(S2, sigma_1, sigma_2, rho) - sigma_adj * m.sqrt(self.T)

    def geometric_basket_option(self, S2, sigma_1, sigma_2, rho, option_type, is_monte=False):

        self.set_seed()
        paths = 100000

        if is_monte:

            Z1 = np.random.normal(0, 1, paths)
            Z2 = np.random.normal(0, 1, paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

            S1_simul = np.zeros(paths)
            S2_simul = np.zeros(paths)
            S1_simul[:] = self.S
            S2_simul[:] = S2

            expo_1 = np.exp((self.r - 0.5 * sigma_1 ** 2) * self.T + sigma_1 * m.sqrt(self.T) * Z1[:])
            expo_2 = np.exp((self.r - 0.5 * sigma_2 ** 2) * self.T + sigma_2 * m.sqrt(self.T) * Z2[:])

            S1_simul = S1_simul[:] * expo_1[:]
            S2_simul = S2_simul[:] * expo_2[:]

            df = np.exp(-self.r * self.T)

            geo_basket_mean = np.sqrt(S1_simul * S2_simul)
            geo_basket_call_payoff = np.maximum(geo_basket_mean - self.K, 0) * df
            geo_basket_put_payoff = np.maximum(self.K - geo_basket_mean, 0) * df
            geo_basket_call_price = np.mean(geo_basket_call_payoff)
            geo_basket_put_price = np.mean(geo_basket_put_payoff)

            if option_type == 'call':
                return geo_basket_call_price
            elif option_type == 'put':
                return geo_basket_put_price
            else:
                return ValueError("Please specify option type!")

        elif not is_monte:
            S_basket = m.sqrt(self.S * S2)  # Geometric mean of the stock prices
            # sigma_adj = Pricer.sigma_basket(sigma_1, sigma_2, rho)
            mu_adj = self.mu_basket(sigma_1, sigma_2, rho)
            d1_basket = self.d1(S2, sigma_1, sigma_2, rho)
            d2_basket = self.d2(S2, sigma_1, sigma_2, rho)

            geometric_basket_call = S_basket * m.exp((mu_adj - self.r) * self.T) * norm.cdf(d1_basket) - self.K * m.exp(
                -self.r * self.T) * norm.cdf(d2_basket)
            geometric_basket_put = self.K * m.exp(-self.r * self.T) * norm.cdf(-d2_basket) - S_basket * m.exp(
                (mu_adj - self.r) * self.T) * norm.cdf(-d1_basket)

            if option_type == 'call':
                return geometric_basket_call
            elif option_type == 'put':
                return geometric_basket_put
            else:
                return ValueError("Please specify option type!")

    def arithmetic_basket_option(self, S2, sigma_1, sigma_2, rho, paths, option_type, with_cv=True):

        self.set_seed()

        Z1 = np.random.normal(0, 1, paths)
        Z2 = np.random.normal(0, 1, paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        S1_simul = np.zeros(paths)
        S2_simul = np.zeros(paths)
        S1_simul[:] = self.S
        S2_simul[:] = S2

        expo_1 = np.exp((self.r - 0.5 * sigma_1 ** 2) * self.T + sigma_1 * m.sqrt(self.T) * Z1[:])
        expo_2 = np.exp((self.r - 0.5 * sigma_2 ** 2) * self.T + sigma_2 * m.sqrt(self.T) * Z2[:])

        S1_simul = S1_simul[:] * expo_1[:]
        S2_simul = S2_simul[:] * expo_2[:]

        df = np.exp(-self.r * self.T)

        arith_basket_mean = (S1_simul + S2_simul) / 2
        arith_basket_call_payoff = np.maximum(arith_basket_mean - self.K, 0) * df
        arith_basket_put_payoff = np.maximum(self.K - arith_basket_mean, 0) * df

        geo_basket_mean = np.sqrt(S1_simul * S2_simul)
        geo_basket_call_payoff = np.maximum(geo_basket_mean - self.K, 0) * df
        geo_basket_put_payoff = np.maximum(self.K - geo_basket_mean, 0) * df

        arith_basket_call_price = np.mean(arith_basket_call_payoff)
        arith_basket_put_price = np.mean(arith_basket_put_payoff)
        geo_basket_call_price = np.mean(geo_basket_call_payoff)
        geo_basket_put_price = np.mean(geo_basket_put_payoff)

        stdev_call = np.std(arith_basket_call_payoff)
        stdev_put = np.std(arith_basket_put_payoff)

        if with_cv:

            covXY_call = np.cov(arith_basket_call_payoff, geo_basket_call_payoff)[0, 1]
            theta_call = covXY_call / np.var(geo_basket_call_payoff)
            covXY_put = np.cov(arith_basket_put_payoff, geo_basket_put_payoff)[0, 1]
            theta_put = covXY_put / np.var(geo_basket_put_payoff)

            adjusted_arith_call_price = arith_basket_call_price + theta_call * (
                    self.geometric_basket_option(S2, sigma_1, sigma_2, rho, option_type='call',
                                                 is_monte=False) - geo_basket_call_price)
            adjusted_arith_put_price = arith_basket_put_price + theta_put * (
                    self.geometric_basket_option(S2, sigma_1, sigma_2, rho, option_type='put',
                                                 is_monte=False) - geo_basket_put_price)

            if option_type == 'call':
                confmc = [adjusted_arith_call_price - 1.96 * stdev_call / m.sqrt(paths), adjusted_arith_call_price + 1.96 * stdev_call / m.sqrt(paths)]
                return adjusted_arith_call_price, confmc
            elif option_type == 'put':
                confmc = [adjusted_arith_put_price - 1.96 * stdev_put / m.sqrt(paths), adjusted_arith_put_price + 1.96 * stdev_put / m.sqrt(paths)]
                return adjusted_arith_put_price, confmc
            else:
                return ValueError("Please specify an option type!")

        elif not with_cv:
            if option_type == 'call':
                confmc = [arith_basket_call_price - 1.96 * stdev_call / m.sqrt(paths),
                          arith_basket_call_price + 1.96 * stdev_call / m.sqrt(paths)]
                return arith_basket_call_price, confmc
            elif option_type == 'put':
                confmc = [arith_basket_put_price - 1.96 * stdev_call / m.sqrt(paths),
                          arith_basket_put_price + 1.96 * stdev_call / m.sqrt(paths)]
                return arith_basket_put_price, confmc
            else:
                return ValueError("Please specify an option type!")

    def kiko_put(self, S, sigma, L, U, N, R):

        self.set_seed()

        deltaT = self.T / N
        M = 16384  # Sobol only supports M with power of 2

        sequencer = qmc.Sobol(d=N, seed=self.seed)
        X = sequencer.random(n=M)
        Z = norm.ppf(X)

        samples = (self.r - 0.5 * sigma ** 2) * deltaT + sigma * m.sqrt(deltaT) * Z
        df_samples = pd.DataFrame(samples)
        df_samples_cumsum = df_samples.cumsum(axis=1)
        df_stocks = S * np.exp(df_samples_cumsum)

        values = []

        for ipath in range(M):

            path = df_stocks.iloc[ipath]
            price_max = path.max()
            price_min = path.min()

            if price_max >= U:
                knockout_time = path[path >= U].index[0]
                payoff = R * np.exp(-self.r * knockout_time * deltaT)
                values.append(payoff)
            elif price_min <= L:
                final_price = path.iloc[-1]
                payoff = max(self.K - final_price, 0) * np.exp(-self.r * self.T)
                values.append(payoff)
            else:  # No knock-out, no knock-in
                values.append(0)

        return np.mean(values)

    def kiko_put_delta(self, S, sigma, L, U, N, R):

        delta_stock = 0.01 * S
        up_val = self.kiko_put(S + delta_stock, sigma, L, U, N, R)
        down_val = self.kiko_put(S - delta_stock, sigma, L, U, N, R)
        delta = (up_val - down_val) / (2 * delta_stock)

        return delta

    def american_option(self, sigma, n, option_type):

        deltaT = self.T / n
        u = m.exp(sigma * m.sqrt(deltaT))
        d = 1 / u
        p = (m.exp(self.r * deltaT) - d) / (u - d)

        stock_price = np.zeros((n + 1, n + 1))
        option_value = np.zeros((n + 1, n + 1))
        stock_price[0][0] = self.S

        for i in range(1, n + 1):
            stock_price[i][0] = stock_price[i - 1][0] * d
            for j in range(1, i + 1):
                stock_price[i][j] = stock_price[i - 1][j - 1] * u

        if option_type == 'call':
            option_value[n] = np.maximum(stock_price[n] - self.K, 0)
        elif option_type == 'put':
            option_value[n] = np.maximum(self.K - stock_price[n], 0)
        else:
            return ValueError("Please specify an option type!")

        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                option_value[i][j] = (p * option_value[i + 1][j + 1] + (1 - p) * option_value[i + 1][j]) * m.exp(
                    -self.r * deltaT)

                if option_type == 'call':
                    option_value[i, j] = np.maximum(option_value[i, j], stock_price[i, j] - self.K)
                else:
                    option_value[i, j] = np.maximum(option_value[i, j], self.K - stock_price[i, j])

        return option_value[0, 0]


# Test Cases:

# Black-Scholes European
# pricer = Pricer(2, 0.03, 0, 3, 2, 100)
# x = pricer.european_option(0.3, 'call')
# print(x)

# Binomial American
# pricer = Pricer(50, 0.1, 0, 2, 70, 100)
# x = pricer.american_option(0.4, 200, 'put')
# print(x)

# Geometric Asian
# pricer = Pricer(100, 0.05, 0, 3, 100, 100)
# x = pricer.asian_geometric_option(0.3, 50, 'call', True)
# print(x)

# Arithmetic Asian
# pricer = Pricer(100, 0.05, 0, 3, 100, 100)
# x = pricer.asian_arithmetic_option(0.3,  50, 100000, 'put', True)
# print(x)

# Geometric Basket
# pricer = Pricer(100, 0.05, 0, 3, 100, 100)
# x = pricer.geometric_basket_option(100, 0.3, 0.3, 0.5, 'call', True)
# print(x)

# Arithmetic Basket
# pricer = Pricer(100, 0.05, 0, 3, 100, 100)
# x = pricer.arithmetic_basket_option(100, 0.3, 0.3,  0.5, 100000, 'call', True)
# print(x)

# Kiko Put
# pricer = Pricer(100, 0.03, 0, 2, 100, 100)
# x = pricer.kiko_put(100, 0.2, 75, 105, 24, 5)
# y = pricer.kiko_put_delta(100, 0.2, 75, 105, 24, 5)
# print(x, y)

# Implied Volatility
# pricer = Pricer(2, 0.03, 0, 3, 2, 100)
# x = pricer.iv(0.48413599739115154, 'call')
# print(x)
