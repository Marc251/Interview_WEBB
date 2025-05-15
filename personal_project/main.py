import logging
import sys
from binance.client import Client
from spot_trading_algo import SpotTradingAlgo, run_live_trading
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the spot trading algorithm')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair to use')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--max_position', type=float, default=0.5, help='Maximum position size as decimal (0.1 = 10%)')
    parser.add_argument('--stop_loss', type=float, default=0.003, help='Stop loss percentage as decimal')
    parser.add_argument('--take_profit', type=float, default=0.005, help='Take profit percentage as decimal')
    parser.add_argument('--interval', type=float, default=1.0, help='Time between iterations in seconds')
    parser.add_argument('--min_spread', type=float, default=0.01, help='Minimum acceptable spread')
    parser.add_argument('--depth', type=int, default=100, help='Order book depth to analyze')
    parser.add_argument('--alpha', type=float, default=0.25, help='Volume imbalance threshold')
    parser.add_argument('--beta', type=float, default=0.25, help='Depth imbalance threshold')
    parser.add_argument('--api_key', type=str, required=False, help='Binance API key')
    parser.add_argument('--api_secret', type=str, required=False, help='Binance API secret')
    parser.add_argument('--maker_fee', type=float, default=0.001, help='Maker fee percentage as decimal (0.001 = 0.1%)') #values for binance
    parser.add_argument('--taker_fee', type=float, default=0.002, help='Taker fee percentage as decimal (0.002 = 0.2%)') #values for binance
    args = parser.parse_args()
    
    #Initialize Binance client
    try:
        client = Client(args.api_key, args.api_secret)
        logging.info("Connected to Binance API")
    except Exception as e:
        logging.error(f"Failed to connect to Binance API: {str(e)}")
        sys.exit(1)
    
    #Logs
    logging.info(f"Starting trading bot with parameters:")
    logging.info(f"Symbol: {args.symbol}")
    logging.info(f"Initial capital: {args.capital}")
    logging.info(f"Stop loss: {args.stop_loss*100}%")
    logging.info(f"Take profit: {args.take_profit*100}%")
    logging.info(f"Max position size: {args.max_position*100}%")
    logging.info(f"Order book depth: {args.depth}")
    logging.info(f"Signal thresholds - Alpha: {args.alpha}, Beta: {args.beta}")
    logging.info(f"Fees - Maker: {args.maker_fee*100}%, Taker: {args.taker_fee*100}%")
    
    algo = SpotTradingAlgo(
        symbol=args.symbol,
        initial_capital=args.capital,
        max_position_size=args.max_position,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        min_spread=args.min_spread,
        n_depth=args.depth,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee
    )
    
    run_live_trading(
        algo=algo, 
        symbol=args.symbol, 
        client=client, 
        interval=args.interval,
        alpha=args.alpha,
        beta=args.beta,
    )