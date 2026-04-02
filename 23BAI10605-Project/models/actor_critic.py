import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared layer (optional, or separate networks)
        self.affine = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head: Outputs probabilities for actions
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic head: Outputs value of the state
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.affine(x)
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values
