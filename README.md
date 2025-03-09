# Blackjack Bot

A sophisticated Deep Q-Learning agent for playing Blackjack with side bets, implementing advanced strategies for bankroll management and risk assessment.

## Features

- Deep Q-Network (DQN) with noisy layers for efficient exploration
- Support for Perfect Pairs and 21+3 side bets
- Advanced card counting and deck penetration tracking
- Dynamic bankroll management with Kelly Criterion
- Risk-adjusted rewards using Sharpe and Sortino ratios
- N-step learning with Prioritized Experience Replay

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
├── src/                    # Source code
│   ├── environment/        # Blackjack environment
│   ├── agents/            # DQN and other agents
│   └── utils/             # Helper utilities
├── tests/                 # Unit tests
├── configs/               # Configuration files
└── notebooks/            # Jupyter notebooks for analysis
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

## Configuration

Training parameters can be modified in `configs/training_config.yaml`.

## Testing

Run the test suite:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 