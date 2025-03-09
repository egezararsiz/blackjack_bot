from collections import deque
import random
import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity (int): Maximum size of buffer
            alpha (float): How much prioritization is used (0 = no prioritization)
            beta_start (float): Initial value of beta for importance sampling
            beta_frames (int): Number of frames over which beta will be annealed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def get_probabilities(self, priority_scale):
        """Get sampling probabilities based on priority."""
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
    
    def get_importance(self, probabilities):
        """Get importance sampling weights."""
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        importance = (1/len(self.buffer) * 1/probabilities) ** beta
        importance_normalized = importance / max(importance)
        return importance_normalized
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(self.alpha)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        
        samples = [self.buffer[idx] for idx in sample_indices]
        importances = self.get_importance(sample_probs[sample_indices])
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (torch.FloatTensor(states), 
                torch.LongTensor(actions), 
                torch.FloatTensor(rewards), 
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones),
                torch.FloatTensor(importances))
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to prevent zero priority
    
    def __len__(self):
        return len(self.buffer)

class NStepReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, n_step=3, gamma=0.99, **kwargs):
        """
        Initialize N-step Replay Buffer.
        
        Args:
            capacity (int): Maximum size of buffer
            n_step (int): Number of steps to look ahead
            gamma (float): Discount factor
            **kwargs: Additional arguments for PrioritizedReplayBuffer
        """
        super().__init__(capacity, **kwargs)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        
    def _get_n_step_info(self):
        """Get n-step reward, next state, and done."""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
            
        return reward, next_state, done
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to n-step buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return
            
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]
        
        super().push(state, action, reward, next_state, done) 