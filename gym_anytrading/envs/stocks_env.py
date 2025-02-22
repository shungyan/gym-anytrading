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
        # Validate index
        start_idx = self.frame_bound[0] - self.window_size
        end_idx = self.frame_bound[1]

        if start_idx < 0:
            raise ValueError("Invalid frame_bound or window_size: index is out of range")

        # Extract slices for all indicators
        open_prices = open_prices[start_idx:end_idx]
        high_prices = high_prices[start_idx:end_idx]
        low_prices = low_prices[start_idx:end_idx]
        close_prices = close_prices[start_idx:end_idx]
        volume = volume[start_idx:end_idx]

        sma_50 = sma_50[start_idx:end_idx]
        ema_50 = ema_50[start_idx:end_idx]
        rsi_14 = rsi_14[start_idx:end_idx]
        macd = macd[start_idx:end_idx]
        macd_signal = macd_signal[start_idx:end_idx]

        stochastic_k = stochastic_k[start_idx:end_idx]
        stochastic_d = stochastic_d[start_idx:end_idx]
        upper_bb = upper_bb[start_idx:end_idx]
        lower_bb = lower_bb[start_idx:end_idx]
        adx = adx[start_idx:end_idx]
        obv = obv[start_idx:end_idx]

        diff = np.insert(np.diff(close_prices), 0, 0)

        signal_features = np.column_stack((open_prices, high_prices, low_prices, close_prices, diff, volume,sma_50, ema_50, rsi_14, macd, macd_signal,stochastic_k, stochastic_d, upper_bb, lower_bb,adx, obv))

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
