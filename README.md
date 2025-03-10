# Blackjack Bot

A sophisticated Deep Q-Learning agent for playing Blackjack, specifically optimized for Pragmatic Play's online blackjack rules, including Perfect Pairs and 21+3 side bets.

## Features

### Game Rules Implementation
- Complete Pragmatic Play Vegas Rules
  - 8-deck shoe with reshuffling at ~50% penetration
  - Dealer stands on all 17s
  - Double on any 2 initial cards (no double after split)
  - Split pairs only (one split per hand)
  - Single card to split aces
  - Insurance when dealer shows ace (pays 2:1)
  - Blackjack pays 3:2
  - Push on tie
  - Multiple spots per player support

### Side Bets
- Perfect Pairs
  - Perfect Pair (25:1)
  - Colored Pair (12:1)
  - Mixed Pair (6:1)
- 21+3 Poker Hands
  - Suited Trips (100:1)
  - Straight Flush (40:1)
  - Three of a Kind (30:1)
  - Straight (10:1)
  - Flush (5:1)

### AI Features
- Deep Q-Network (DQN) with noisy layers
- Advanced card counting and penetration tracking
- Kelly Criterion for optimal bet sizing
- Risk-adjusted rewards using:
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
- N-step learning with Prioritized Experience Replay
- Multi-spot strategy optimization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blackjack_bot.git
cd blackjack_bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e ".[dev]"
```

## Project Structure

```
blackjack_bot/
├── src/
│   ├── environment/        # Blackjack environment with Pragmatic Play rules
│   │   ├── blackjack_env.py
│   │   ├── card.py
│   │   └── enums.py
│   ├── agents/            # AI agents
│   │   ├── dqn_agent.py
│   │   └── networks.py
│   └── utils/             # Helper utilities
│       ├── replay_buffer.py
│       └── risk_calculator.py
├── tests/                 # Comprehensive test suite
├── configs/               # Configuration files
└── notebooks/            # Analysis notebooks
```

## Usage

1. Train the agent:
```bash
python -m src.train
```

2. Monitor training progress:
```bash
tensorboard --logdir runs/
```

3. Analyze performance:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Configuration

Training parameters can be modified in `configs/training_config.yaml`:
- Environment settings (deck count, bankroll, etc.)
- Network architecture
- Training hyperparameters
- Risk management settings

## Performance Metrics

The agent tracks:
- Win/loss ratio
- Return on investment (ROI)
- Sharpe and Sortino ratios
- Maximum drawdown
- Side bet success rates
- Card counting efficiency

## Testing

Run the complete test suite:
```bash
pytest tests/ --cov=src
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run pre-commit hooks (`pre-commit install`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This project is for educational purposes only. Please check your local regulations regarding the use of AI systems in gambling environments. 