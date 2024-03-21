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


class Mid(IStrategy):
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
    buy_bb_period = IntParameter(5, 60, default=15, space='buy', optimize=True)
    buy_supertrend_period = IntParameter(5, 60, default=15, space='buy', optimize=True)
    buy_supertrend_multiplier = DecimalParameter(0.5, 3, default=1, decimals=1, space='buy', optimize=True)
    buy_bb_width_value = DecimalParameter(0.02, 0.2, default=0.02, decimals=2, space='buy', optimize=True)
    buy_add_lost_pct = DecimalParameter(0.01, 0.1, default=0.05, decimals=2, space='buy', optimize=True)

    tpsl_atr_period = IntParameter(5, 60, default=15, space='buy', optimize=True)
    atr_sl_rate = DecimalParameter(0.3, 3, default=0.3, decimals=1, space='buy', optimize=True)
    tpsl_rate = DecimalParameter(0.3, 2.6, default=0.3, decimals=2, space='buy', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # buy indicators
        bb_bands = pta.bbands(dataframe['close'], length=self.buy_bb_period.value)
        dataframe['bb_lowerband'] = bb_bands.iloc[:, 0]
        dataframe['bb_upperband'] = bb_bands.iloc[:, 2]
        dataframe['bb_middleband'] = bb_bands.iloc[:, 1]
        dataframe['bb_width'] = bb_bands.iloc[:, 3]

        superT = pta.supertrend(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=self.buy_supertrend_period.value, multiplier=self.buy_supertrend_multiplier.value)
        dataframe['supertrend'] = superT.iloc[:, 1]

        atr = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.tpsl_atr_period.value)
        dataframe['atr'] = atr

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        buy_1 = (
            (dataframe['close'] < dataframe['bb_middleband']) &
            (dataframe['supertrend'] == 1) &
            (dataframe['bb_width'] > self.buy_bb_width_value.value)
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
                        
        # if current_time - timedelta(minutes=10) < trade.open_date_utc:
        #     if current_profit >= 0.05:
        #         return "profit_sell_fast"

        # if current_profit > 0:
        #     if current_candle["fastk"] > self.sell_fastx.value:
        #         return "fastk_profit_sell"

        #     if current_candle["cci"] > self.sell_cci.value:
        #         return "cci_profit_sell"

        # if current_time - timedelta(hours=2) > trade.open_date_utc:
        #     if current_profit > 0:
        #         return "profit_sell_in_2h"
                
        # if current_candle["high"] >= trade.open_rate:
        #     if current_candle["cci"] > self.sell_cci.value:
        #         return "cci_sell"

        # if current_profit > self.sell_loss_cci_profit.value:
        #     if current_candle["cci"] > self.sell_loss_cci.value:
        #         return "cci_loss_sell"
        # if current_profit > 0.05:
            # print(dir(trade))
            # ['_LocalTrade__set_stop_loss', '__annotations__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slotnames__', '__str__', '__subclasshook__', '__weakref__', '_calc_base_close', '_calc_open_trade_value', '_date_last_filled_utc', 'add_bt_trade', 'adjust_min_max_rates', 'adjust_stop_loss', 'amount', 'amount_precision', 'amount_requested', 'base_currency', 'borrowed', 'bt_open_open_trade_count', 'bt_trades_open_pp', 'buy_tag', 'calc_close_trade_value', 'calc_profit', 'calc_profit_ratio', 'calculate_interest', 'calculate_profit', 'close', 'close_bt_trade', 'close_date', 'close_date_utc', 'close_profit', 'close_profit_abs', 'close_rate', 'close_rate_requested', 'contract_size', 'date_last_filled_utc', 'enter_tag', 'entry_side', 'exchange', 'exit_order_status', 'exit_reason', 'exit_side', 'fee_close', 'fee_close_cost', 'fee_close_currency', 'fee_open', 'fee_open_cost', 'fee_open_currency', 'fee_updated', 'from_json', 'funding_fee_running', 'funding_fees', 'get_canceled_exit_order_count', 'get_open_trade_count', 'get_open_trades', 'get_trades_proxy', 'has_no_leverage', 'has_open_orders', 'id', 'initial_stop_loss', 'initial_stop_loss_pct', 'interest_rate', 'is_open', 'is_short', 'is_stop_loss_trailing', 'leverage', 'liquidation_price', 'max_rate', 'max_stake_amount', 'min_rate', 'nr_of_successful_buys', 'nr_of_successful_entries', 'nr_of_successful_exits', 'nr_of_successful_sells', 'open_date', 'open_date_utc', 'open_orders', 'open_orders_ids', 'open_rate', 'open_rate_requested', 'open_trade_value', 'orders', 'pair', 'precision_mode', 'price_precision', 'realized_profit', 'recalc_open_trade_value', 'recalc_trade_from_orders', 'remove_bt_trade', 'reset_trades', 'safe_base_currency', 'safe_close_rate', 'safe_quote_currency', 'select_filled_or_open_orders', 'select_filled_orders', 'select_order', 'select_order_by_order_id', 'sell_reason', 'set_funding_fees', 'set_liquidation_price', 'stake_amount', 'stake_currency', 'stop_loss', 'stop_loss_pct', 'stoploss_last_update', 'stoploss_last_update_utc', 'stoploss_or_liquidation', 'stoploss_order_id', 'stoploss_reinitialization', 'strategy', 'timeframe', 'to_json', 'total_profit', 'trade_direction', 'trades', 'trades_open', 'trading_mode', 'update_fee', 'update_order', 'update_trade', 'use_db']
            # return "profit_sell
            # "
        if current_profit < 0:
            time_diff = current_time - trade.open_date_utc
            delay_bar = int(time_diff.total_seconds() / (60*15))
            open_candle = dataframe.iloc[-delay_bar] if delay_bar != 0 else current_candle
            if current_candle["close"] < (open_candle["close"]  - open_candle['atr'] * self.atr_sl_rate.value):
                return "lost_sell"
        
        if current_profit > 0:
            time_diff = current_time - trade.open_date_utc
            delay_bar = int(time_diff.total_seconds() / (60*15))
            open_candle = dataframe.iloc[-delay_bar] if delay_bar != 0 else current_candle
            if current_candle["close"] > (open_candle["close"] + open_candle['atr'] * self.tpsl_rate.value):
                return "profit_sell"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe