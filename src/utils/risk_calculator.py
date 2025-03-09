from collections import deque
import numpy as np

class RiskCalculator:
    def __init__(self, risk_free_rate=0.02, max_leverage=2.0, window_size=100):
        """
        Initialize the risk calculator.
        
        Args:
            risk_free_rate (float): Annual risk-free rate
            max_leverage (float): Maximum allowed leverage
            window_size (int): Size of the rolling window for calculations
        """
        self.risk_free_rate = risk_free_rate / 365  # Convert to daily rate
        self.max_leverage = max_leverage
        self.returns_history = deque(maxlen=window_size)
        self.metrics_history = {
            'sharpe_ratio': deque(maxlen=window_size),
            'sortino_ratio': deque(maxlen=window_size),
            'max_drawdown': deque(maxlen=window_size),
            'win_rate': deque(maxlen=window_size)
        }

    def calculate_metrics(self):
        """Calculate various risk metrics"""
        if len(self.returns_history) < 2:
            return {metric: 0.0 for metric in self.metrics_history.keys()}

        returns = np.array(self.returns_history)
        excess_returns = returns - self.risk_free_rate
        
        # Sharpe Ratio
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-6)
        
        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        sortino = np.mean(excess_returns) / (np.std(downside_returns) if len(downside_returns) > 0 else 1e-6)
        
        # Maximum Drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-6)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win Rate
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        
        metrics = {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        
        # Update history
        for metric, value in metrics.items():
            self.metrics_history[metric].append(value)
            
        return metrics

    def calculate_kelly_fraction(self, win_rate, win_loss_ratio):
        """
        Calculate the optimal Kelly Criterion fraction.
        
        Args:
            win_rate (float): Probability of winning
            win_loss_ratio (float): Ratio of average win to average loss
            
        Returns:
            float: Optimal fraction of bankroll to bet
        """
        q = 1 - win_rate
        if q == 0 or win_loss_ratio == 0:
            return 0
        kelly = win_rate - (q / win_loss_ratio)
        return max(0, min(kelly, self.max_leverage))

    def adjust_reward(self, reward, bankroll, initial_bankroll):
        """
        Adjust the reward based on risk metrics.
        
        Args:
            reward (float): Original reward
            bankroll (float): Current bankroll
            initial_bankroll (float): Initial bankroll
            
        Returns:
            float: Risk-adjusted reward
        """
        # Calculate return for this step
        returns = (bankroll - initial_bankroll) / initial_bankroll
        self.returns_history.append(returns)
        
        # Calculate risk metrics
        metrics = self.calculate_metrics()
        
        # Risk adjustment factor based on Sharpe and Sortino ratios
        risk_adjustment = (metrics['sharpe_ratio'] + metrics['sortino_ratio']) / 2
        
        # Apply dynamic leverage based on performance
        if metrics['sharpe_ratio'] > 1.0 and metrics['sortino_ratio'] > 1.0:
            leverage = min(
                self.max_leverage,
                1.0 + (metrics['sharpe_ratio'] + metrics['sortino_ratio']) / 4
            )
        else:
            leverage = 1.0
            
        # Adjust reward
        adjusted_reward = reward * (1 + risk_adjustment) * leverage
        
        return adjusted_reward

    def get_metrics_summary(self):
        """Get a summary of recent risk metrics"""
        return {
            metric: np.mean(list(values)) if values else 0.0
            for metric, values in self.metrics_history.items()
        } 