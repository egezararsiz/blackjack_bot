import pytest
import torch
import numpy as np
from src.agents.dqn_agent import DQNAgent

@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        'agent': {
            'network': {
                'use_dueling': False,
                'use_noisy_nets': True,
                'hidden_sizes': [64, 32],
                'dropout_rate': 0.1,
                'initial_lr': 0.001,
                'lr_min': 0.0001,
                'lr_t0': 100,
                'lr_tmult': 2
            },
            'training': {
                'batch_size': 32,
                'gamma': 0.99,
                'target_update': 10
            }
        },
        'replay_buffer': {
            'capacity': 1000,
            'n_step': 3,
            'alpha': 0.6,
            'beta_start': 0.4,
            'beta_frames': 1000
        },
        'risk_adjustment': {
            'risk_free_rate': 0.02,
            'max_leverage': 2.0
        }
    }

@pytest.fixture
def agent(config):
    """Create a test agent."""
    return DQNAgent(state_dim=100, action_dim=5, config=config)

def test_agent_initialization(agent, config):
    """Test agent initialization."""
    assert isinstance(agent.policy_net, torch.nn.Module)
    assert isinstance(agent.target_net, torch.nn.Module)
    assert agent.batch_size == config['agent']['training']['batch_size']
    assert agent.gamma == config['agent']['training']['gamma']
    assert agent.target_update == config['agent']['training']['target_update']

def test_select_action(agent):
    """Test action selection."""
    state = np.random.random(100)
    action = agent.select_action(state)
    assert isinstance(action, int)
    assert 0 <= action < 5

def test_update(agent):
    """Test agent update."""
    # Create fake batch
    state = np.random.random(100)
    action = 0
    reward = 1.0
    next_state = np.random.random(100)
    done = False
    bankroll = 10000
    initial_bankroll = 10000
    
    # First update should return 0 (not enough samples)
    loss = agent.update(state, action, reward, next_state, done, bankroll, initial_bankroll)
    assert loss == 0.0
    
    # Fill buffer with samples
    for _ in range(agent.batch_size):
        agent.update(state, action, reward, next_state, done, bankroll, initial_bankroll)
    
    # Now should return non-zero loss
    loss = agent.update(state, action, reward, next_state, done, bankroll, initial_bankroll)
    assert loss > 0.0

def test_update_metrics(agent):
    """Test metrics update."""
    initial_reward = agent.running_reward
    episode_reward = 100.0
    
    agent.update_metrics(episode_reward)
    
    assert len(agent.episode_rewards) == 1
    assert agent.running_reward != initial_reward
    assert agent.best_reward == max(initial_reward, agent.running_reward)

def test_save_load(agent, tmp_path):
    """Test model saving and loading."""
    # Save model
    save_path = tmp_path / "test_model.pth"
    agent.save(str(save_path))
    
    # Create new agent
    new_agent = DQNAgent(state_dim=100, action_dim=5, config=agent._config)
    
    # Load saved model
    new_agent.load(str(save_path))
    
    # Check if states match
    assert torch.equal(
        next(agent.policy_net.parameters()),
        next(new_agent.policy_net.parameters())
    )
    assert agent.training_step == new_agent.training_step
    assert agent.running_reward == new_agent.running_reward
    assert agent.best_reward == new_agent.best_reward

def test_get_metrics(agent):
    """Test metrics retrieval."""
    metrics = agent.get_metrics()
    
    assert isinstance(metrics, dict)
    assert 'training_step' in metrics
    assert 'running_reward' in metrics
    assert 'best_reward' in metrics
    assert 'recent_rewards' in metrics
    assert 'learning_rate' in metrics
    assert 'risk_metrics' in metrics

def test_noise_reset(agent):
    """Test noise reset in noisy networks."""
    if agent.policy_net.noisy:
        # Get initial parameters
        initial_params = next(agent.policy_net.parameters()).clone()
        
        # Reset noise
        agent.policy_net.reset_noise()
        
        # Parameters should be different after reset
        assert not torch.equal(initial_params, next(agent.policy_net.parameters()))

def test_target_net_update(agent):
    """Test target network update."""
    # Get initial target parameters
    initial_target_params = next(agent.target_net.parameters()).clone()
    
    # Update policy net
    state = np.random.random(100)
    action = 0
    reward = 1.0
    next_state = np.random.random(100)
    done = False
    bankroll = 10000
    initial_bankroll = 10000
    
    # Fill buffer and perform updates
    for _ in range(agent.batch_size + agent.target_update):
        agent.update(state, action, reward, next_state, done, bankroll, initial_bankroll)
    
    # Target parameters should be different now
    assert not torch.equal(initial_target_params, next(agent.target_net.parameters())) 