This jupyter notebook details the implementation of a stock trading reinforcement learning (RL) agent using a linear Q-network to decide among buy/sell/hold actions for three stocks (AAPL, MSI, SBUX). The project simulates a trading environment and compares the RL agent's performance against baseline strategies (buy-and-hold and random trading).

Key components include:

-> Data Preparation: Stock prices are normalized to prevent instability during training.
-> Trading Environment: The MyStockTradingEnv class models the trading process, allowing 27 possible action combinations (3 actions × 3 stocks) and returning a 7-dimensional state vector (shares held, current prices, and cash).
-> Q-Network: A linear model approximates the Q-values Q(s,a)=Ws+b, updated via gradient descent with momentum and L2 regularization. A separate target network stabilizes learning.
-> Replay Buffer: The PrioritizedReplayBuffer samples experiences based on their importance (TD error), improving learning efficiency.
-> Training Loop: The agent is trained over multiple episodes using epsilon-greedy exploration, updating weights and target networks periodically.
-> Evaluation and Comparison: The trained model is evaluated on unseen data and benchmarked against buy-and-hold and random strategies, with portfolio values and returns analyzed visually and statistically.