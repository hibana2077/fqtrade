"""
PEDHyperOptLoss

This module is a custom HyperoptLoss class

PED = Profit, Expectancy, Duration weighted somewaht equally

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""
from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import numpy as np
from typing import Any, Dict


# Contstants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration
EXPECTED_TRADES_PER_DAY = 3                         # used to set target goals
MIN_TRADES_PER_DAY = EXPECTED_TRADES_PER_DAY / 2    # used to filter out scenarios where there are not enough trades
EXPECTED_PROFIT_PER_TRADE = 0.010                   # be realistic. Setting this too high will eliminate potentially good solutions
EXPECTED_AVE_PROFIT = 0.050                         # used to assess actual profit vs desired profit. OK to set high
EXPECTED_MONTHLY_PROFIT = 0.15                      # used to assess actual profit vs desired profit. Typical is 15% (0.15)
EXPECTED_TRADE_DURATION = 3.0*60.0*60.0             # goal for duration (or shorter) in seconds
MAX_TRADE_DURATION = 10.0*60.0*60.0                 # max allowable duration (in seconds)

# Win/Loss-specific constants
MIN_WINLOSS_RATIO = 1.0
EXPECTED_WINLOSS_RATIO = 2.0

UNDESIRED_SOLUTION = 2.0             # indicates that we don't want this solution (so hyperopt will avoid)


class PEDHyperOptLoss(IHyperOptLoss):


    """
    Defines a custom loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        debug_level = 0 # displays (more) messages if higher

        # define weights
        weight_num_trades = 0.6
        weight_duration = 2.0
        weight_abs_profit = 1.0
        weight_exp_profit = 0.2
        weight_ave_profit = 0.2
        weight_expectancy = 2.0
        weight_win_loss_ratio = 1.0
        weight_sharp_ratio = 1.5
        weight_sortino_ratio = 0.25
        weight_drawdown = 1.0
        weight_profit_approx = 0.0

        if config['exchange']['name']:
            if (config['exchange']['name'] == 'kucoin') or (config['exchange']['name'] == 'ascendex'):
                # kucoin is extremely volatile, with v.high profits in backtesting (but not in real markets)
                # so, reduce influence of absolute profit and no. of trades (and sharpe/sortino)
                # the goal is reduce the number of losing and highly risky trades (the cost is some loss of profits)
                weight_num_trades = 0.1
                weight_duration = 2.0
                weight_abs_profit = 0.1
                weight_exp_profit = 0.2
                weight_ave_profit = 0.2
                weight_expectancy = 3.0
                weight_win_loss_ratio = 2.0
                weight_sharp_ratio = 0.25
                weight_sortino_ratio = 0.05


        days_period = (max_date - min_date).days
        # target_trades = days_period*EXPECTED_TRADES_PER_DAY
        if config['max_open_trades']:
            target_trades = days_period * config['max_open_trades']
        else:
            target_trades = days_period * EXPECTED_TRADES_PER_DAY

        # Calculate trade loss metric first, because this is used elsewhere
        # Several other metrics are misleading if there are not enough trades

        # trade loss
        if trade_count > MIN_TRADES_PER_DAY * days_period:
            num_trades_loss = (target_trades - trade_count) / target_trades
        else:
            # just return a large number if insufficient trades. Makes other calculations easier/safer
            if debug_level > 1:
                print(" \tTrade count too low:{:.0f}".format(trade_count))
            return UNDESIRED_SOLUTION

        # Absolute Profit
        num_months = max((days_period / 30.0), 1.0)
        if backtest_stats['profit_total_abs']:
            profit_sum = backtest_stats['profit_total_abs']
        else:
            profit_sum = results["profit_abs"].sum()

        if profit_sum < 0.0:
            if debug_level > 2:
                print(" \tProfit too low: {:.2f}".format(profit_sum))
            if backtest_stats['profit_total']:
                return 1.0 - backtest_stats['profit_total']
            else:
                return UNDESIRED_SOLUTION

        if backtest_stats['profit_total']:
            abs_profit_loss = backtest_stats['profit_total']
        elif config['dry_run_wallet']:
            abs_profit_loss = profit_sum / config['dry_run_wallet']
        elif config['max_open_trades'] and config['stake_amount']:
            abs_profit_loss = profit_sum / (config['max_open_trades'] * config['stake_amount'] * num_months)
        else:
            abs_profit_loss = profit_sum / 10000.0

        # scale loss by #months so that it's consistent no matter the length of the run
        # use 15% per month as goal, scale by 10
        abs_profit_loss = 10.0 * (0.15 - (abs_profit_loss / num_months))

        # # punish if below goal
        # if abs_profit_loss > 0.0:
        #     if debug_level > 1:
        #         print(" \tProfit loss below goal: {:.2f}".format(abs_profit_loss))
        #     return abs_profit_loss

        # Daily/Average profit
        if backtest_stats['profit_mean']:
            ave_profit = backtest_stats['profit_mean']
        else:
            ave_profit = ((profit_sum / days_period) / 100.0)
        ave_profit_loss = EXPECTED_AVE_PROFIT - ave_profit
        ave_profit_loss = ave_profit_loss * 100.0

        # Expected Profit

        # note that we don't have enough info to calculate profit % because we don't know the original investment
        # so, we approximate
        total_profit = results["profit_abs"]

        if backtest_stats['starting_balance']:
            expected_sum = backtest_stats['starting_balance'] * (1.0 + EXPECTED_MONTHLY_PROFIT * num_months)
        else:
            expected_sum = results['stake_amount'].mean() * trade_count * EXPECTED_PROFIT_PER_TRADE
        exp_profit_loss = (expected_sum - profit_sum) / expected_sum

        # if num_trades_loss < 0.0:
        #     print("profit_sum:{:.2f} expected_sum:{:.2f} ave_profit_loss:{:.2f} exp_profit_loss:{:.2f}" \
        #           .format(profit_sum, expected_sum, ave_profit_loss, exp_profit_loss))

        # trade duration (taken from default loss function)
        trade_duration = results['trade_duration'].mean()
        duration_loss = (trade_duration - EXPECTED_TRADE_DURATION) / EXPECTED_TRADE_DURATION

        # punish if below goal
        if trade_duration > MAX_TRADE_DURATION:
            if debug_level > 1:
                print(" \tTrade duration below goal: {:.2f}".format(trade_duration))
            return UNDESIRED_SOLUTION

        # Winning trades
        results['upside_returns'] = 0
        results.loc[total_profit > 0.0001, 'upside_returns'] = 1.0

        if backtest_stats['wins']:
            winning_count = backtest_stats['wins']
        else:
            winning_count = results['upside_returns'].sum()

        # Losing trades
        results['downside_returns'] = 0
        results.loc[total_profit < 0, 'downside_returns'] = 1.0
        losing_count = trade_count - winning_count

        if backtest_stats['losses']:
            act_losing_count = backtest_stats['wins']
        else:
            act_losing_count = results['downside_returns'].sum()


        # if winning_count < (2.0 * losing_count):
        if winning_count < (1.0 * losing_count):
            if debug_level > 1:
                print(" \tWinning count below goal: {:.0f} vs {:.0f}".format(winning_count, losing_count))
            return UNDESIRED_SOLUTION

        # Expectancy (refer to freqtrade edge page for info)
        w = winning_count / trade_count
        l = 1.0 - w
        results['net_gain'] = results['profit_abs'] * results['upside_returns']
        results['net_loss'] = results['profit_abs'] * results['downside_returns']
        ave_profit = results['net_gain'].sum() / trade_count
        ave_loss = results['net_loss'].sum() / trade_count
        if abs(ave_loss) < 0.001:
            ave_loss = 0.001
        r = ave_profit / abs(ave_loss)
        e = r * w - l

        expectancy_loss = -e
        if expectancy_loss > 0.0:
            if debug_level > 1:
                print(" \tExpectancy Loss below goal: {:.2f}".format(expectancy_loss))
            return expectancy_loss

        # Win/Loss ratio (losses here are draws & losses)
        win_loss_ratio_loss = 0.0
        if losing_count > 0:
            win_loss_ratio_loss = 1.0 - (winning_count / losing_count)
        else:
            win_loss_ratio_loss = -abs(abs_profit_loss)

        # # punish if below goal
        # if win_loss_ratio_loss > 0.0:
        #     if debug_level > 1:
        #         print(" \tWin/Loss Ratio below goal: {:.2f}".format(win_loss_ratio_loss))
        #     return UNDESIRED_SOLUTION

        # Sharpe Ratio
        expected_returns_mean = total_profit.sum() / days_period
        up_stdev = np.std(total_profit)
        if up_stdev != 0:
            # calculate Sharpe ratio, but scale down to match other parameters
            sharp_ratio_loss = 0.01 - (expected_returns_mean / up_stdev * np.sqrt(365)) / 100.0
        else:
            if debug_level > 1:
                print(" \tSharp ratio below goal")
            return UNDESIRED_SOLUTION

        # Sortino Ratio
        down_stdev = np.std(results['downside_returns'])
        if down_stdev != 0:
            sortino_ratio_loss = -1.0 * (expected_returns_mean / down_stdev * np.sqrt(365)) / 10000.0
        else:
            if debug_level > 1:
                print(" \tSortino ratio below goal")
            return UNDESIRED_SOLUTION

        # amplify if both Sharpe and Sortino are -ve or both +ve
        if ((sharp_ratio_loss < 0.0) and (sortino_ratio_loss < 0.0)) or \
                ((sharp_ratio_loss > 0.0) and (sortino_ratio_loss > 0.0)):
            sharp_ratio_loss = 2.0 * sharp_ratio_loss
            sortino_ratio_loss = 2.0 * sortino_ratio_loss

        # Max Drawdown
        drawdown_loss = 0.0
        if backtest_stats['max_drawdown']:
            drawdown_loss = (backtest_stats['max_drawdown'] - 1.0)

        # Approximate Profit
        if backtest_stats['stoploss']:
            stoploss = abs(backtest_stats['stoploss'])
        else:
            stoploss = abs(ave_loss)

        if stoploss > 0.9: # custom stoploss probably used
            stoploss = max(abs(ave_loss), 0.1)

        profit_approx_loss = ( (winning_count * ave_profit) - (act_losing_count * stoploss) ) / days_period / 100.0
        profit_approx_loss = EXPECTED_AVE_PROFIT - profit_approx_loss

        # weight the results (values are based on trial & error). Goal is for anything -ve to be a decent  solution
        num_trades_loss = weight_num_trades * num_trades_loss
        duration_loss = weight_duration * duration_loss
        abs_profit_loss = weight_abs_profit * abs_profit_loss
        exp_profit_loss = weight_exp_profit * exp_profit_loss
        ave_profit_loss = weight_ave_profit * ave_profit_loss
        expectancy_loss = weight_expectancy * expectancy_loss
        win_loss_ratio_loss = weight_win_loss_ratio * win_loss_ratio_loss
        sharp_ratio_loss = weight_sharp_ratio * sharp_ratio_loss
        sortino_ratio_loss = weight_sortino_ratio * sortino_ratio_loss
        drawdown_loss = weight_drawdown * drawdown_loss
        profit_approx_loss = weight_profit_approx * profit_approx_loss

        if weight_abs_profit > 0.0:
            # sometimes spikes happen, so cap it and turn on debug
            if abs_profit_loss < -20.0:
                abs_profit_loss = max(abs_profit_loss, -20.0)
                debug_level = 1

        # don't let anything outweigh profit
        if weight_num_trades > 0.0:
            num_trades_loss = max(num_trades_loss, abs_profit_loss)
        if weight_duration > 0.0:
            duration_loss = max(duration_loss, abs_profit_loss)
        if weight_exp_profit > 0.0:
            exp_profit_loss = max(exp_profit_loss, abs_profit_loss)
        if weight_ave_profit > 0.0:
            ave_profit_loss = max(ave_profit_loss, abs_profit_loss)
        if weight_expectancy > 0.0:
            expectancy_loss = max(expectancy_loss, abs_profit_loss)
        if weight_win_loss_ratio > 0.0:
            win_loss_ratio_loss = max(win_loss_ratio_loss, abs_profit_loss)
        if weight_sharp_ratio > 0.0:
            sharp_ratio_loss = max(sharp_ratio_loss, abs_profit_loss)
        if weight_sortino_ratio > 0.0:
            sortino_ratio_loss = max(sortino_ratio_loss, abs_profit_loss)
        if weight_drawdown > 0.0:
            drawdown_loss = max(drawdown_loss, abs_profit_loss)

        result = abs_profit_loss + num_trades_loss + duration_loss + exp_profit_loss + ave_profit_loss + \
                 win_loss_ratio_loss + expectancy_loss + sharp_ratio_loss + sortino_ratio_loss + drawdown_loss + \
                 profit_approx_loss

        if (abs_profit_loss < 0.0) & (result < 0.0) and (debug_level > 0):
            print(" \tPabs:{:.2f} Pave:{:.2f} n:{:.2f} dur:{:.2f} w/l:{:.2f} " \
              "expy:{:.2f}  sharpe:{:.2f} sortino:{:.2f} draw:{:.2f} Papr:{:.2f}" \
              " Total:{:.2f}" \
              .format(abs_profit_loss, ave_profit_loss, num_trades_loss, duration_loss, win_loss_ratio_loss, \
                      expectancy_loss, sharp_ratio_loss, sortino_ratio_loss, drawdown_loss, profit_approx_loss, \
                      result))

        return result