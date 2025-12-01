import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Flexible MLP for DQN.
    - state_dim: input size (CartPole: 4)
    - action_dim: output size (CartPole: 2)
    - hidden_dims: list of hidden layer sizes, e.g.:
        []              -> linear (no hidden layer)
        [128]           -> 2-layer net (1 hidden + output)
        [128,128,128,128] -> 5-layer net (4 hidden + output)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128]

        layers = []
        in_dim = state_dim

        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            in_dim = h

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through all hidden layers with ReLU
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # Output layer (no activation, Q-values)
        x = self.output_layer(x)
        return x