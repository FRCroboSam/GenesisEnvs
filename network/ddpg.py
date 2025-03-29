import torch.nn as nn
import torch

# need to bound actor values iwth tanh
class DDPG_Actor(nn.Module):
    #here input dim should be: input space + action
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DDPG_Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            
        )

    def forward(self, x):
        return torch.tanh(self.network(x))
    
    
# function approximator lerning Q value , dont restrict it using tanh 
class DDPG_Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DDPG_Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)