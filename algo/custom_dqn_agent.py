import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network.dqn import DQN
from .replay_buffer import ReplayBuffer

class CustomDQNAgent:
    def __init__(self, input_dim, output_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min, device, load=False, num_envs=1, hidden_dim=64, checkpoint_path=None, batch_size=None, replay_size=None):
        self.device = device
        self.num_envs = num_envs
        self.model = DQN(input_dim, output_dim, hidden_dim).to(self.device)
        self.target_model = DQN(input_dim, output_dim, hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.checkpoint_path = checkpoint_path
        if load: 
            self.load_checkpoint()
            print("Loaded model from checkpoint")
        self.use_double = False
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon if not load else 0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.memory = ReplayBuffer(replay_size, state_dim=input_dim, action_dim=1, device=self.device)
    
    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.model.eval()
        self.target_model.eval()
        print(f"Checkpoint loaded from {self.checkpoint_path}")

    def select_action(self, state):
        with torch.no_grad():
            q_values = self.model(state)
        #epsilon greedy selection:
        #   calls torch.rand num_envs times (which envs should be random)
        #   actions has shape (num_envs, num_possible actions)
        #   choose random or greedy actions based on the mask
        num_envs = q_values.size(0)
        random_action = torch.randint(0, q_values.size(1), (num_envs,)).to(self.device)
        greedy_action = torch.argmax(q_values, dim=1)
        mask = (torch.rand(num_envs) < self.epsilon).to(self.device)
        actions = torch.where(mask, random_action, greedy_action)
        return actions

    def train(self):
        if self.memory.size < self.batch_size: # 1 -> 64
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        #Q values have shape(batch_size, output_dim) (batch size is 64 * num_envs)
        q_values = self.model(states) #critic on current state
        qa_values = self.target_model(states) # target critic on current state
        next_qa_values = self.target_model(next_states) #target critic q values on next states
        target_values = q_values.clone()  #placeholder ot make sure shape is correct
        
        with torch.no_grad():
            if self.use_double:  #target_model = target_critic, model = critic
                next_action = torch.argmax(q_values, dim=1)
                # Evaluate the Q-value of the optimal action using the target network
                
                #evaluate how good next_action is with target network
                next_q_values = qa_values[torch.arange(next_qa_values.size(0)), next_action]  # Shape: (batch_size,)
                
                # estimated maximum possible reward from taking current action, best future action
                target_values[torch.arange(self.batch_size), actions.squeeze().to(torch.int64)] \
                    = rewards.squeeze() + self.gamma * next_q_values * (1 - dones)
            else:
                target_values[torch.arange(target_values.size(0)), actions.squeeze().to(torch.int64)] \
                    = rewards.squeeze() + self.gamma * torch.max(next_q_values, 1)[0] * (1 - dones)

            loss = nn.MSELoss()(q_values, target_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
