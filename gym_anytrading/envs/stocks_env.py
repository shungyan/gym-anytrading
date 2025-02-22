import numpy as np

from .trading_env import TradingEnv, Actions, Positions


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _process_data(self):
        open_prices = self.df.loc[:, "Open"].to_numpy()
        high_prices = self.df.loc[:, "High"].to_numpy()
        low_prices = self.df.loc[:, "Low"].to_numpy()
        close_prices = self.df.loc[:, "Close"].to_numpy()
        volume = self.df.loc[:, "Volume"].to_numpy()

        sma_50 = self.df.loc[:, "SMA_50"].to_numpy()
        ema_50 = self.df.loc[:, "EMA_50"].to_numpy()
        rsi_14 = self.df.loc[:, "RSI_14"].to_numpy()
        macd = self.df.loc[:, "MACD"].to_numpy()
        macd_signal = self.df.loc[:, "MACD_Signal"].to_numpy()

        stochastic_k = self.df.loc[:, "Stochastic_K"].to_numpy()
        stochastic_d = self.df.loc[:, "Stochastic_D"].to_numpy()
        upper_bb = self.df.loc[:, "Upper_BB"].to_numpy()
        lower_bb = self.df.loc[:, "Lower_BB"].to_numpy()
        adx = self.df.loc[:, "ADX"].to_numpy()
        obv = self.df.loc[:, "OBV"].to_numpy()

        close_prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        close_prices = close_prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(close_prices), 0, 0)

        signal_features = np.column_stack((open_prices, high_prices, low_prices, close_prices, volume, diff,sma_50, ema_50, rsi_14, macd, macd_signal,stochastic_k, stochastic_d, upper_bb, lower_bb,adx, obv))

        return close_prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

        return step_reward

    def _update_profit(self, action):
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
