import pytest
import numpy as np
import torch
from src.utils.replay_buffer import PrioritizedReplayBuffer, NStepReplayBuffer
from src.utils.risk_calculator import RiskCalculator

@pytest.fixture
def replay_buffer():
    """Create a test replay buffer."""
    return PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=1000
    )

@pytest.fixture
def n_step_buffer():
    """Create a test n-step replay buffer."""
    return NStepReplayBuffer(
        capacity=100,
        n_step=3,
        gamma=0.99,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=1000
    )

@pytest.fixture
def risk_calculator():
    """Create a test risk calculator."""
    return RiskCalculator(
        risk_free_rate=0.02,
        max_leverage=2.0,
        window_size=100
    )

def test_replay_buffer_push(replay_buffer):
    """Test pushing transitions to buffer."""
    state = np.random.random(10)
    action = 0
    reward = 1.0
    next_state = np.random.random(10)
    done = False
    
    replay_buffer.push(state, action, reward, next_state, done)
    
    assert len(replay_buffer) == 1
    assert len(replay_buffer.priorities) == 1

def test_replay_buffer_sample(replay_buffer):
    """Test sampling from buffer."""
    # Fill buffer
    for _ in range(10):
        state = np.random.random(10)
        action = 0
        reward = 1.0
        next_state = np.random.random(10)
        done = False
        replay_buffer.push(state, action, reward, next_state, done)
    
    batch_size = 5
    states, actions, rewards, next_states, dones, importances = replay_buffer.sample(batch_size)
    
    assert isinstance(states, torch.FloatTensor)
    assert isinstance(actions, torch.LongTensor)
    assert isinstance(rewards, torch.FloatTensor)
    assert isinstance(next_states, torch.FloatTensor)
    assert isinstance(dones, torch.FloatTensor)
    assert isinstance(importances, torch.FloatTensor)
    
    assert states.shape[0] == batch_size
    assert actions.shape[0] == batch_size
    assert rewards.shape[0] == batch_size
    assert next_states.shape[0] == batch_size
    assert dones.shape[0] == batch_size
    assert importances.shape[0] == batch_size

def test_n_step_buffer_push(n_step_buffer):
    """Test pushing transitions to n-step buffer."""
    state = np.random.random(10)
    action = 0
    reward = 1.0
    next_state = np.random.random(10)
    done = False
    
    # First n-1 pushes should not add to main buffer
    for _ in range(n_step_buffer.n_step - 1):
        n_step_buffer.push(state, action, reward, next_state, done)
        assert len(n_step_buffer) == 0
    
    # nth push should add to main buffer
    n_step_buffer.push(state, action, reward, next_state, done)
    assert len(n_step_buffer) == 1

def test_n_step_reward_calculation(n_step_buffer):
    """Test n-step reward calculation."""
    state = np.random.random(10)
    action = 0
    next_state = np.random.random(10)
    done = False
    
    # Push n transitions with rewards 1, 2, 3
    for i in range(n_step_buffer.n_step):
        n_step_buffer.push(state, action, float(i + 1), next_state, done)
    
    # Sample and check if reward is properly discounted
    states, actions, rewards, next_states, dones, _ = n_step_buffer.sample(1)
    expected_reward = 1 + n_step_buffer.gamma * 2 + n_step_buffer.gamma**2 * 3
    assert abs(rewards[0].item() - expected_reward) < 1e-5

def test_risk_calculator_metrics(risk_calculator):
    """Test risk calculator metrics."""
    # Add some returns
    returns = [0.01, -0.02, 0.03, -0.01, 0.02]
    for ret in returns:
        risk_calculator.returns_history.append(ret)
    
    metrics = risk_calculator.calculate_metrics()
    
    assert 'sharpe_ratio' in metrics
    assert 'sortino_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics

def test_risk_calculator_kelly(risk_calculator):
    """Test Kelly Criterion calculation."""
    win_rate = 0.6
    win_loss_ratio = 2.0
    
    kelly = risk_calculator.calculate_kelly_fraction(win_rate, win_loss_ratio)
    
    assert 0 <= kelly <= risk_calculator.max_leverage

def test_risk_calculator_reward_adjustment(risk_calculator):
    """Test reward adjustment."""
    reward = 100.0
    bankroll = 11000
    initial_bankroll = 10000
    
    # Add some history for meaningful metrics
    for _ in range(10):
        risk_calculator.returns_history.append(0.01)
    
    adjusted_reward = risk_calculator.adjust_reward(reward, bankroll, initial_bankroll)
    
    assert isinstance(adjusted_reward, float)
    assert adjusted_reward != reward  # Should be adjusted based on metrics

def test_risk_calculator_metrics_summary(risk_calculator):
    """Test metrics summary."""
    # Add some returns
    for _ in range(10):
        risk_calculator.returns_history.append(0.01)
    
    # Calculate metrics to populate history
    risk_calculator.calculate_metrics()
    
    summary = risk_calculator.get_metrics_summary()
    
    assert isinstance(summary, dict)
    assert 'sharpe_ratio' in summary
    assert 'sortino_ratio' in summary
    assert 'max_drawdown' in summary
    assert 'win_rate' in summary 