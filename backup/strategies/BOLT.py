from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)


class BOLT(IStrategy):
    minimal_roi = {
        "0": 1
    }
    timeframe = '15m'
    process_only_new_candles = True
    startup_candle_count = 120
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }
    stoploss = -0.25

    is_optimize_32 = True

    buy_ma_period = IntParameter(5, 60, default=15, space='buy', optimize=True)

    sell_fastx = IntParameter(50, 100, default=70, space='sell', optimize=True)
    sell_loss_cci = IntParameter(low=0, high=600, default=148, space='sell', optimize=False)
    sell_loss_cci_profit = DecimalParameter(-0.15, 0, default=-0.04, decimals=2, space='sell', optimize=False)
    sell_cci = IntParameter(low=0, high=200, default=90, space='sell', optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # buy indicators
        dataframe['hammer'] = pta.cdl_pattern(open_=dataframe['open'], high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], name='hammer')
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.buy_ma_period.value)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        buy_1 = (
            (dataframe['hammer'] == 100) &
            (dataframe['close'] > dataframe['sma'])
        )
        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
                        
        if current_time - timedelta(minutes=10) < trade.open_date_utc:
            if current_profit >= 0.05:
                return "profit_sell_fast"

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"

            if current_candle["cci"] > self.sell_cci.value:
                return "cci_profit_sell"

        if current_time - timedelta(hours=2) > trade.open_date_utc:
            if current_profit > 0:
                return "profit_sell_in_2h"
                
        if current_candle["high"] >= trade.open_rate:
            if current_candle["cci"] > self.sell_cci.value:
                return "cci_sell"

        if current_profit > self.sell_loss_cci_profit.value:
            if current_candle["cci"] > self.sell_loss_cci.value:
                return "cci_loss_sell"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe