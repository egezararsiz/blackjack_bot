import pytest
import torch
import numpy as np
from src.agents.dqn_agent import DQNAgent

@pytest.fixture
def input_dims():
    """Create test input dimensions."""
    return {
        'rank_frequencies': 13,
        'counts': 3,
        'hand_info': 4,
        'bets': 4
    }

@pytest.fixture
def sample_observation(input_dims):
    """Create a sample observation dictionary."""
    return {
        'rank_frequencies': np.random.randint(0, 4, input_dims['rank_frequencies']).astype(np.float32),
        'counts': np.random.normal(0, 1, input_dims['counts']).astype(np.float32),
        'hand_info': np.random.random(input_dims['hand_info']).astype(np.float32),
        'bets': np.random.random(input_dims['bets']).astype(np.float32)
    }

@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        'agent': {
            'network': {
                'use_dueling': True,
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
def agent(input_dims, config):
    """Create a test agent."""
    return DQNAgent(input_dims=input_dims, action_dim=5, config=config)

def test_agent_initialization(agent, config, input_dims):
    """Test agent initialization."""
    assert isinstance(agent.policy_net, torch.nn.Module)
    assert isinstance(agent.target_net, torch.nn.Module)
    assert agent.batch_size == config['agent']['training']['batch_size']
    assert agent.gamma == config['agent']['training']['gamma']
    assert agent.target_update == config['agent']['training']['target_update']
    
    # Test network architecture
    for net in [agent.policy_net, agent.target_net]:
        assert net.rank_freq_dim == input_dims['rank_frequencies']
        assert net.counts_dim == input_dims['counts']
        assert net.hand_info_dim == input_dims['hand_info']
        assert net.bets_dim == input_dims['bets']

def test_prepare_state(agent, sample_observation):
    """Test state preparation."""
    state_tensors = agent._prepare_state(sample_observation)
    
    for key in sample_observation:
        assert isinstance(state_tensors[key], torch.Tensor)
        assert state_tensors[key].shape == (1, sample_observation[key].shape[0])
        assert state_tensors[key].device == agent.device

def test_select_action(agent, sample_observation):
    """Test action selection."""
    action = agent.select_action(sample_observation)
    assert isinstance(action, int)
    assert 0 <= action < 5

def test_update(agent, sample_observation):
    """Test agent update."""
    # Create fake batch
    state = sample_observation
    action = 0
    reward = 1.0
    next_state = {k: v + 0.1 for k, v in sample_observation.items()}
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

def test_save_load(agent, tmp_path, sample_observation):
    """Test model saving and loading."""
    # Get initial action and Q-values
    initial_action = agent.select_action(sample_observation)
    with torch.no_grad():
        initial_q_values = agent.policy_net(agent._prepare_state(sample_observation))
    
    # Save model
    save_path = tmp_path / "test_model.pth"
    agent.save(str(save_path))
    
    # Create new agent
    new_agent = DQNAgent(
        input_dims=agent.policy_net.rank_freq_dim,
        action_dim=5,
        config=agent._config
    )
    
    # Load saved model
    new_agent.load(str(save_path))
    
    # Check if actions and Q-values match
    loaded_action = new_agent.select_action(sample_observation)
    with torch.no_grad():
        loaded_q_values = new_agent.policy_net(new_agent._prepare_state(sample_observation))
    
    assert initial_action == loaded_action
    assert torch.allclose(initial_q_values, loaded_q_values)
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
        initial_params = {}
        for name, param in agent.policy_net.named_parameters():
            if 'weight_mu' in name:
                initial_params[name] = param.clone()
        
        # Reset noise
        agent.policy_net.reset_noise()
        
        # Parameters should be different after reset
        for name, param in agent.policy_net.named_parameters():
            if 'weight_mu' in name:
                assert not torch.equal(initial_params[name], param)

def test_target_net_update(agent, sample_observation):
    """Test target network update."""
    # Get initial target parameters
    initial_target_params = {}
    for name, param in agent.target_net.named_parameters():
        if 'weight_mu' in name:
            initial_target_params[name] = param.clone()
    
    # Update policy net
    state = sample_observation
    action = 0
    reward = 1.0
    next_state = {k: v + 0.1 for k, v in sample_observation.items()}
    done = False
    bankroll = 10000
    initial_bankroll = 10000
    
    # Fill buffer and perform updates
    for _ in range(agent.batch_size + agent.target_update):
        agent.update(state, action, reward, next_state, done, bankroll, initial_bankroll)
    
    # Target parameters should be different now
    for name, param in agent.target_net.named_parameters():
        if 'weight_mu' in name:
            assert not torch.equal(initial_target_params[name], param) 