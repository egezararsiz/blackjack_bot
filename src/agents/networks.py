import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        """
        Initialize a noisy linear layer.
        
        Args:
            in_features (int): Input features
            out_features (int): Output features
            sigma_init (float): Initial noise standard deviation
        """
        super(NoisyLinear, self).__init__()
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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 128], noisy=True, dropout_rate=0.2):
        """
        Initialize DQN network.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension (number of actions)
            hidden_sizes (list): List of hidden layer sizes
            noisy (bool): Whether to use noisy linear layers
            dropout_rate (float): Dropout probability
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers
        for size in hidden_sizes:
            if noisy:
                layers.append(NoisyLinear(prev_size, size))
            else:
                layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        # Output layer
        if noisy:
            layers.append(NoisyLinear(prev_size, output_dim))
        else:
            layers.append(nn.Linear(prev_size, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.noisy = noisy
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
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
        """Reset noise for all noisy layers."""
        if self.noisy:
            for module in self.net:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def forward(self, x):
        """Forward pass."""
        return self.net(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 128], noisy=True, dropout_rate=0.2):
        """
        Initialize Dueling DQN network.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension (number of actions)
            hidden_sizes (list): List of hidden layer sizes
            noisy (bool): Whether to use noisy linear layers
            dropout_rate (float): Dropout probability
        """
        super(DuelingDQN, self).__init__()
        
        # Feature layer
        feature_layers = []
        prev_size = input_dim
        for size in hidden_sizes[:-1]:  # All but last hidden layer
            if noisy:
                feature_layers.append(NoisyLinear(prev_size, size))
            else:
                feature_layers.append(nn.Linear(prev_size, size))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Value stream
        value_layers = []
        if noisy:
            value_layers.append(NoisyLinear(prev_size, hidden_sizes[-1]))
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Dropout(dropout_rate))
            value_layers.append(NoisyLinear(hidden_sizes[-1], 1))
        else:
            value_layers.append(nn.Linear(prev_size, hidden_sizes[-1]))
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Dropout(dropout_rate))
            value_layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.value_stream = nn.Sequential(*value_layers)
        
        # Advantage stream
        advantage_layers = []
        if noisy:
            advantage_layers.append(NoisyLinear(prev_size, hidden_sizes[-1]))
            advantage_layers.append(nn.ReLU())
            advantage_layers.append(nn.Dropout(dropout_rate))
            advantage_layers.append(NoisyLinear(hidden_sizes[-1], output_dim))
        else:
            advantage_layers.append(nn.Linear(prev_size, hidden_sizes[-1]))
            advantage_layers.append(nn.ReLU())
            advantage_layers.append(nn.Dropout(dropout_rate))
            advantage_layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        
        self.advantage_stream = nn.Sequential(*advantage_layers)
        
        self.noisy = noisy
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
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
        """Reset noise for all noisy layers."""
        if self.noisy:
            for module in [self.feature_layer, self.value_stream, self.advantage_stream]:
                for layer in module:
                    if isinstance(layer, NoisyLinear):
                        layer.reset_noise()

    def forward(self, x):
        """Forward pass with dueling architecture."""
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        return value + (advantage - advantage.mean(dim=1, keepdim=True)) 