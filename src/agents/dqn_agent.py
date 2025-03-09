import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Any

from .networks import DQN, DuelingDQN
from ..utils.replay_buffer import NStepReplayBuffer
from ..utils.risk_calculator import RiskCalculator

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any]
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            config (dict): Configuration dictionary
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        
        # Load network configuration
        network_config = config['agent']['network']
        self.use_dueling = network_config.get('use_dueling', False)
        
        # Initialize networks
        NetworkClass = DuelingDQN if self.use_dueling else DQN
        self.policy_net = NetworkClass(
            state_dim,
            action_dim,
            hidden_sizes=network_config['hidden_sizes'],
            noisy=network_config['use_noisy_nets'],
            dropout_rate=network_config['dropout_rate']
        ).to(self.device)
        
        self.target_net = NetworkClass(
            state_dim,
            action_dim,
            hidden_sizes=network_config['hidden_sizes'],
            noisy=network_config['use_noisy_nets'],
            dropout_rate=network_config['dropout_rate']
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer with learning rate scheduler
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=network_config['initial_lr']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=network_config['lr_t0'],
            T_mult=network_config['lr_tmult'],
            eta_min=network_config['lr_min']
        )
        
        # Initialize replay buffer
        buffer_config = config['replay_buffer']
        self.replay_buffer = NStepReplayBuffer(
            capacity=buffer_config['capacity'],
            n_step=buffer_config['n_step'],
            gamma=config['agent']['training']['gamma'],
            alpha=buffer_config['alpha'],
            beta_start=buffer_config['beta_start'],
            beta_frames=buffer_config['beta_frames']
        )
        
        # Initialize risk calculator
        risk_config = config['risk_adjustment']
        self.risk_calculator = RiskCalculator(
            risk_free_rate=risk_config['risk_free_rate'],
            max_leverage=risk_config['max_leverage']
        )
        
        # Training parameters
        train_config = config['agent']['training']
        self.batch_size = train_config['batch_size']
        self.gamma = train_config['gamma']
        self.target_update = train_config['target_update']
        
        # Metrics
        self.training_step = 0
        self.episode_rewards = []
        self.running_reward = 0
        self.best_reward = float('-inf')
        
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using the policy network.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        bankroll: float,
        initial_bankroll: float
    ) -> float:
        """
        Update the agent's knowledge.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
            bankroll (float): Current bankroll
            initial_bankroll (float): Initial bankroll
            
        Returns:
            float: Loss value
        """
        # Apply risk adjustment to reward
        adjusted_reward = self.risk_calculator.adjust_reward(reward, bankroll, initial_bankroll)
        
        # Store transition in replay buffer
        self.replay_buffer.push(state, action, adjusted_reward, next_state, done)
        
        # Return if not enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch and prepare tensors
        states, actions, rewards, next_states, dones, importances = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        importances = importances.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using double Q-learning
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss with importance sampling weights
        loss = (importances.unsqueeze(1) * (current_q_values - target_q_values) ** 2).mean()
        
        # Update priorities in replay buffer
        td_errors = (current_q_values - target_q_values).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(range(self.batch_size), td_errors)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network if needed
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update learning rate
        self.scheduler.step()
        
        return loss.item()
    
    def update_metrics(self, episode_reward: float) -> None:
        """
        Update training metrics.
        
        Args:
            episode_reward (float): Total reward for the episode
        """
        self.episode_rewards.append(episode_reward)
        self.running_reward = 0.05 * episode_reward + 0.95 * self.running_reward
        
        if self.running_reward > self.best_reward:
            self.best_reward = self.running_reward
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'training_step': self.training_step,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'replay_buffer': self.replay_buffer,
            'running_reward': self.running_reward,
            'best_reward': self.best_reward,
            'episode_rewards': self.episode_rewards,
            'risk_calculator': self.risk_calculator
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.training_step = checkpoint['training_step']
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.replay_buffer = checkpoint['replay_buffer']
        self.running_reward = checkpoint['running_reward']
        self.best_reward = checkpoint['best_reward']
        self.episode_rewards = checkpoint['episode_rewards']
        self.risk_calculator = checkpoint['risk_calculator']
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current training metrics.
        
        Returns:
            dict: Dictionary containing current metrics
        """
        return {
            'training_step': self.training_step,
            'running_reward': self.running_reward,
            'best_reward': self.best_reward,
            'recent_rewards': self.episode_rewards[-100:],
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'risk_metrics': self.risk_calculator.get_metrics_summary()
        } 