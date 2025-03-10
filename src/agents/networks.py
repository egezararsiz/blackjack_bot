import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset learnable parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Scale noise for the factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Reset the factorized Gaussian noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.epsilon_weight = epsilon_out.ger(epsilon_in)
        self.epsilon_bias = epsilon_out

    def forward(self, x):
        """Forward pass with noise."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.epsilon_weight
            bias = self.bias_mu + self.bias_sigma * self.epsilon_bias
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class ModularDQN(nn.Module):
    """DQN with separate processing paths for different observation components."""
    def __init__(self, input_dims, output_dim, hidden_sizes=[256, 128], noisy=True, dropout_rate=0.2):
        super().__init__()
        
        # Input dimensions for each component
        self.rank_freq_dim = input_dims['rank_frequencies']
        self.counts_dim = input_dims['counts']
        self.hand_info_dim = input_dims['hand_info']
        self.bets_dim = input_dims['bets']
        
        LinearLayer = NoisyLinear if noisy else nn.Linear
        
        # Rank frequencies processing path
        self.rank_freq_net = nn.Sequential(
            LinearLayer(self.rank_freq_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Counts processing path
        self.counts_net = nn.Sequential(
            LinearLayer(self.counts_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Hand info processing path
        self.hand_info_net = nn.Sequential(
            LinearLayer(self.hand_info_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Bets processing path
        self.bets_net = nn.Sequential(
            LinearLayer(self.bets_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined features dimension
        combined_dim = 32 + 16 + 16 + 16  # Sum of output dimensions from each path
        
        # Main network
        layers = []
        prev_size = combined_dim
        
        for size in hidden_sizes:
            layers.append(LinearLayer(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        layers.append(LinearLayer(prev_size, output_dim))
        
        self.main_net = nn.Sequential(*layers)
        self.noisy = noisy
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, NoisyLinear)):
            if hasattr(module, 'weight_mu'):
                nn.init.xavier_uniform_(module.weight_mu)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if hasattr(module, 'bias_mu'):
                    nn.init.zeros_(module.bias_mu)
                else:
                    nn.init.zeros_(module.bias)
    
    def reset_noise(self):
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
    def forward(self, x):
        # Process each component
        rank_freq_features = self.rank_freq_net(x['rank_frequencies'])
        counts_features = self.counts_net(x['counts'])
        hand_info_features = self.hand_info_net(x['hand_info'])
        bets_features = self.bets_net(x['bets'])
        
        # Combine features
        combined = torch.cat([
            rank_freq_features,
            counts_features,
            hand_info_features,
            bets_features
        ], dim=1)
        
        return self.main_net(combined)

class ModularDuelingDQN(nn.Module):
    """Dueling DQN with separate processing paths for different observation components."""
    def __init__(self, input_dims, output_dim, hidden_sizes=[256, 128], noisy=True, dropout_rate=0.2):
        super().__init__()
        
        # Input dimensions for each component
        self.rank_freq_dim = input_dims['rank_frequencies']
        self.counts_dim = input_dims['counts']
        self.hand_info_dim = input_dims['hand_info']
        self.bets_dim = input_dims['bets']
        
        LinearLayer = NoisyLinear if noisy else nn.Linear
        
        # Component processing paths (same as ModularDQN)
        self.rank_freq_net = nn.Sequential(
            LinearLayer(self.rank_freq_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.counts_net = nn.Sequential(
            LinearLayer(self.counts_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.hand_info_net = nn.Sequential(
            LinearLayer(self.hand_info_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.bets_net = nn.Sequential(
            LinearLayer(self.bets_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined features dimension
        combined_dim = 32 + 16 + 16 + 16
        
        # Feature layer
        feature_layers = []
        prev_size = combined_dim
        for size in hidden_sizes[:-1]:
            feature_layers.append(LinearLayer(prev_size, size))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            LinearLayer(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            LinearLayer(hidden_sizes[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            LinearLayer(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            LinearLayer(hidden_sizes[-1], output_dim)
        )
        
        self.noisy = noisy
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, NoisyLinear)):
            if hasattr(module, 'weight_mu'):
                nn.init.xavier_uniform_(module.weight_mu)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if hasattr(module, 'bias_mu'):
                    nn.init.zeros_(module.bias_mu)
                else:
                    nn.init.zeros_(module.bias)
    
    def reset_noise(self):
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
    def forward(self, x):
        # Process each component
        rank_freq_features = self.rank_freq_net(x['rank_frequencies'])
        counts_features = self.counts_net(x['counts'])
        hand_info_features = self.hand_info_net(x['hand_info'])
        bets_features = self.bets_net(x['bets'])
        
        # Combine features
        combined = torch.cat([
            rank_freq_features,
            counts_features,
            hand_info_features,
            bets_features
        ], dim=1)
        
        # Process through feature layer
        features = self.feature_layer(combined)
        
        # Calculate value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        return value + (advantage - advantage.mean(dim=1, keepdim=True)) 