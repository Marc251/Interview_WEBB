import pandas as pd
import requests
import time
import logging
import sys
from datetime import datetime
from typing import Dict, Tuple, Any



def get_order_book(ticker: str, client) -> pd.DataFrame:
    """
    Gets the live order book for the specified ticker.
    Returns a DataFrame with bid and ask prices and volumes.
    """
    order_book = client.get_order_book(symbol=ticker)    
    bids = pd.DataFrame(order_book['bids'])
    bids = bids.rename(columns={0: "bid", 1: "volume"})
    bids['ask'] = float('nan')
    asks = pd.DataFrame(order_book['asks'])
    asks = asks.rename(columns={0: "ask", 1: "volume"}) 
    asks['bid'] = float('nan')
    book = pd.concat([bids, asks]).reset_index(drop=True)
    book = book.astype({"bid": float, "ask": float, "volume": float})
    book = book.sort_values(
        by=['bid', 'ask'], 
        ascending=[False, True], 
        na_position='last'
    ).reset_index(drop=True)
    return book


def get_current_price(ticker: str) -> float:
    """Get the current price of a ticker from Binance API"""
    response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={ticker}")
    if response.status_code != 200:
        raise Exception(f"Error fetching price: {response.text}")
    return float(response.json()["price"])


class SpotTradingAlgo:
    def __init__(self, 
                 symbol: str,
                 initial_capital: float,
                 max_position_size: float,
                 stop_loss_pct: float,
                 take_profit_pct: float,
                 min_spread: float,
                 n_depth: int,
                 maker_fee: float,  # 0.1% default maker fee
                 taker_fee: float): # 0.1% default taker fee
        """
        Initialize the trading algorithm with parameters.
        Args:
            initial_capital: Starting capital amount
            max_position_size: Maximum position size as a percentage of capital (0.0-1.0)
            stop_loss_pct: Stop loss percentage (0.0-1.0)
            take_profit_pct: Take profit percentage (0.0-1.0)
            min_spread: Minimum acceptable spread to consider asset liquid
            n_depth: Number of order book levels to analyze
            maker_fee: Fee percentage for maker orders (0.0-1.0)
            taker_fee: Fee percentage for taker orders (0.0-1.0)
        """
        self.symbol=symbol
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_spread = min_spread
        self.n_depth = n_depth
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.position = 0
        self.entry_price = 0
        self.trades_history = []
        self.total_fees_paid = 0.0
        self.performance_metrics = {
            'win_rate': 0,
            'avg_profit': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'profitable_trades': 0,
            'total_fees': 0.0
        }

    def analyze_order_book(self, order_book: pd.DataFrame) -> Dict[str, float]:
        """
        Analyzes order book to determine market depth and potential pressure.
        
        Returns:
            Dict with analysis metrics:
            - imbalance: Ratio of buy vs sell orders (>0 means more buyers)
            - depth_imbalance: Monetary imbalance (>0 means more money on buy side)
            - spread: Percentage difference between best bid and ask
        """
        bid_mask = order_book["bid"].notna()
        ask_mask = order_book["ask"].notna()
        best_bid = order_book.loc[bid_mask, "bid"].max() if bid_mask.any() else None
        best_ask = order_book.loc[ask_mask, "ask"].min() if ask_mask.any() else None
        if best_bid is None or best_ask is None:
            raise ValueError("No valid bids or asks in order book.")
        # Calculate spread
        spread = (best_ask - best_bid) / best_bid
        # Get volumes up to n_depth levels
        bid_df = order_book[bid_mask].sort_values("bid", ascending=False).head(self.n_depth)
        ask_df = order_book[ask_mask].sort_values("ask", ascending=True).head(self.n_depth)
        # Calculate total volume imbalance
        bid_vol = bid_df["volume"].sum()
        ask_vol = ask_df["volume"].sum()
        if bid_vol + ask_vol > 0:
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        else:
            imbalance = 0
        #Calculate weighted depth (price × volume)
        bid_depth = (bid_df["bid"] * bid_df["volume"]).sum()
        ask_depth = (ask_df["ask"] * ask_df["volume"]).sum()
        #Calculate depth imbalance
        if bid_depth + ask_depth > 0:
            depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        else:
            depth_imbalance = 0
        return {
            'spread': float(spread),
            'imbalance': float(imbalance),
            'depth_imbalance': float(depth_imbalance),
            'best_bid': float(best_bid),
            'best_ask': float(best_ask),
            'bid_depth': float(bid_depth),
            'ask_depth': float(ask_depth)
        }

    def calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on current capital and max position size parameter.
        Returns the quantity to buy/sell.
        """
        return self.capital * self.max_position_size / price

    def should_open_position(self, 
                           order_book_analysis: Dict[str, float],
                           alpha: float,
                           beta: float) -> Tuple[bool, str]:
        """
        Determine if we should open a position based on order book analysis.
        Args:
            order_book_analysis: Dict containing order book metrics
            alpha: Threshold for volume imbalance
            beta: Threshold for depth imbalance
        Returns:
            Tuple of (should_trade: bool, direction: str)
        """
        # Check if spread is acceptable (liquid enough)
        if order_book_analysis['spread'] > self.min_spread:
            return False, ""
        # Calculate expected fees
        expected_fee_pct = self.taker_fee * 2  # Both entry and exit fees
        # For a buy signal, we need significant buying pressure AND enough imbalance to overcome fees
        if (order_book_analysis['imbalance'] > alpha and 
            order_book_analysis["depth_imbalance"] > beta and
            order_book_analysis["depth_imbalance"] > expected_fee_pct * 2):  # Must have at least 2x fees in edge
            return True, "buy"
        # For a sell signal, we need significant selling pressure AND enough imbalance to overcome fees
        elif (order_book_analysis['imbalance'] < -alpha and 
              order_book_analysis["depth_imbalance"] < -beta and
              order_book_analysis["depth_imbalance"] < -expected_fee_pct * 2):  # Must have at least 2x fees in edge
            return True, "sell"
        # No clear signal or not enough edge to overcome fees: we wait
        else:
            return False, ''

    def should_close_position(self, current_price: float) -> Tuple[bool, float]:
        """
        Check if we should close position based on stop loss/take profit.
        Returns:
            Tuple of (should_close: bool, pnl_percentage: float)
        """
        if self.position == 0:
            return False, 0
        # Calculate percentage PnL
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        # Adjust for short positions
        if self.position < 0:
            pnl_pct = -pnl_pct
        # Check if stop loss or take profit triggered
        if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
            return True, pnl_pct
        return False, pnl_pct

    def execute_trade(self, 
                     current_price: float, 
                     order_book: pd.DataFrame, 
                     alpha:float,
                     beta: float) -> Dict[str, Any]:
        """
        Main trading logic that processes order book and executes trades accordingly.
        Args:
            current_price: Current market price
            order_book: Order book DataFrame
        Returns:
            Dict with trade details
        """
        # Analyze order book
        order_book_analysis = self.analyze_order_book(order_book)
        # Initialize trade details
        trade_details = {
            'timestamp': datetime.now(),
            'action': 'none',
            'price': current_price,
            'size': 0,
            'reason': '',
            'pnl': 0,
            'fees': 0,
            'capital': self.capital
        }
        # First check if we should close an existing position
        should_close, pnl_pct = self.should_close_position(current_price)
        if should_close:
            trade_details['action'] = 'close'
            trade_details['size'] = abs(self.position)
            # Determine if it's stop loss or take profit
            if pnl_pct >= self.take_profit_pct:
                trade_details['reason'] = 'take_profit'
            else:
                trade_details['reason'] = 'stop_loss'
            # Calculate fees on the closing transaction (using taker fee)
            closing_value = abs(self.position) * current_price
            fee_amount = closing_value * self.taker_fee
            self.total_fees_paid += fee_amount
            # Calculate actual PnL after fees
            pnl_amount = self.position * (current_price - self.entry_price)
            net_pnl_amount = pnl_amount - fee_amount
            net_pnl_pct = net_pnl_amount / (abs(self.position) * self.entry_price)
            # Update trade details
            trade_details['pnl'] = net_pnl_pct  # Net PnL percentage after fees
            trade_details['gross_pnl'] = pnl_pct  # Gross PnL before fees
            trade_details['fees'] = fee_amount
            # Update capital based on net PnL
            self.capital += net_pnl_amount
            # Reset position
            self.position = 0
            self.entry_price = 0
            # Update performance metrics
            self._update_metrics(net_pnl_pct > 0, net_pnl_pct, fee_amount)
            # Save trade to history
            self.trades_history.append(trade_details)
            return trade_details
        # Check if we should open a new position
        should_trade, direction = self.should_open_position(
            order_book_analysis, 
            alpha=alpha,
            beta=beta
        )
        if should_trade and self.position == 0:
            position_size = self.calculate_position_size(current_price)
            # Calculate fees for the opening transaction (using taker fee)
            opening_value = position_size * current_price
            fee_amount = opening_value * self.taker_fee
            # Adjust position size to account for fees
            # This ensures we stay within our max position size constraint
            adjusted_position_size = position_size * (1 - self.taker_fee)
            # Update position and fees
            if direction == "buy":
                self.position = adjusted_position_size
                trade_details['action'] = 'buy'
            else:
                self.position = -adjusted_position_size
                trade_details['action'] = 'sell'
            self.entry_price = current_price
            trade_details['size'] = abs(adjusted_position_size)
            trade_details['reason'] = f'order_book_imbalance_{direction}'
            trade_details['fees'] = fee_amount
            # Update fees paid and capital
            self.total_fees_paid += fee_amount
            self.capital -= fee_amount  # Subtract fees from capital
            # Update performance metrics
            self._update_metrics(False, 0, fee_amount)
            # Save trade to history
            self.trades_history.append(trade_details)
        return trade_details
    
    def _update_metrics(self, is_profitable: bool, pnl_pct: float, fee_amount: float = 0.0) -> None:
        """Update performance metrics after a trade"""
        self.performance_metrics['total_trades'] += 1
        if is_profitable:
            self.performance_metrics['profitable_trades'] += 1
        #Update win rate
        self.performance_metrics['win_rate'] = (
            self.performance_metrics['profitable_trades'] / 
            self.performance_metrics['total_trades']
        )
        #Update average profit
        prev_avg = self.performance_metrics['avg_profit']
        total_trades = self.performance_metrics['total_trades']
        self.performance_metrics['avg_profit'] = (
            (prev_avg * (total_trades - 1) + pnl_pct) / total_trades
        )
        #Update max drawdown
        current_drawdown = (self.capital - self.initial_capital) / self.initial_capital
        self.performance_metrics['max_drawdown'] = min(
            self.performance_metrics['max_drawdown'],
            current_drawdown
        )
        # Update total fees
        self.performance_metrics['total_fees'] += fee_amount
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the algorithm's performance"""
        current_price = 0
        try:
            current_price = get_current_price(self.symbol)
        except:
            pass
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'total_fees_paid': self.total_fees_paid,
            'fee_impact': self.total_fees_paid / self.initial_capital if self.initial_capital > 0 else 0,
            'metrics': self.performance_metrics,
            'total_trades': len(self.trades_history),
            'active_position': {
                'size': self.position,
                'entry_price': self.entry_price,
                'current_pnl': self.should_close_position(
                    current_price
                )[1] if self.position != 0 and current_price > 0 else 0
            }
        }


def run_live_trading(algo: SpotTradingAlgo, symbol: str, client, alpha: float, beta: float, interval: float = 1.0):
    """
    Run the trading algorithm in an infinite loop
    Args:
        algo: Instance of SpotTradingAlgo class
        symbol: Trading ticker
        client: API client for the exchange
        interval: Time between iterations in seconds (default: 1 second)
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Starting live trading for {symbol}")
    logging.info(f"Initial capital: {algo.capital}")
    logging.info(f"Trading parameters: Stop loss: {algo.stop_loss_pct*100}%, Take profit: {algo.take_profit_pct*100}%")
    trade_count = 0
    profitable_trades = 0
    
    try:
        while True:
            try:
                # Get current market data
                order_book = get_order_book(symbol, client)
                current_price = get_current_price(symbol)
                # Analyze order book
                trade = algo.execute_trade(
                    current_price=current_price, 
                    order_book=order_book,
                    alpha=alpha,
                    beta=beta
                )
                #Log trade if action taken
                if trade['action'] != 'none':
                    trade_count += 1
                    if trade['action'] == 'close' and trade['pnl'] > 0:
                        profitable_trades += 1
                    #Log trade details
                    logging.info(f"Trade #{trade_count}: {trade['action']} {trade['size']} @ {trade['price']}")
                    logging.info(f"Reason: {trade['reason']}")
                    if trade['action'] == 'close':
                        win_rate = profitable_trades / trade_count if trade_count > 0 else 0
                        logging.info(f"Trade P&L: {trade['pnl']*100:.5f}%")
                        logging.info(f"Win rate: {win_rate*100:.2f}%")
                    #Calculate current P&L
                    if algo.initial_capital > 0:
                        pnl = 100 * (algo.capital - algo.initial_capital) / algo.initial_capital
                        logging.info(f"Current capital: {algo.capital:.2f}")
                        logging.info(f"Overall P&L: {pnl:.5f}%")
                #If we have an open position, log current unrealized P&L
                if algo.position != 0:
                    _, unrealized_pnl = algo.should_close_position(current_price)
                    logging.info(f"Position: {algo.position} @ {algo.entry_price}, Unrealized P&L: {unrealized_pnl*100:.5f}%")
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Error in trading loop: {str(e)}")
                time.sleep(interval)  # Continue the loop after an error
            
    except KeyboardInterrupt:
        logging.info("Trading stopped by user")
        # Log final statistics
        final_pnl = 100 * (algo.capital - algo.initial_capital) / algo.initial_capital if algo.initial_capital > 0 else 0
        win_rate = profitable_trades / trade_count if trade_count > 0 else 0
        logging.info(f"Final capital: {algo.capital:.2f}")
        logging.info(f"Final P&L: {final_pnl:.5f}%")
        logging.info(f"Total trades: {trade_count}")
        logging.info(f"Profitable trades: {profitable_trades}")
        logging.info(f"Win rate: {win_rate*100:.2f}%")

        return {
            'final_capital': algo.capital,
            'pnl_percent': final_pnl,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'trade_history': algo.trades_history
        }