import torch
import numpy as np
import random


class HER_ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, goal_dim, device, her_ratio=0.8, reward_fn=None):
        """
        Hindsight Experience Replay Buffer
        
        Args:
            max_size (int): Maximum number of transitions to store
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            goal_dim (int): Dimension of goal space
            device (torch.device): Device to store tensors on
            her_ratio (float): Ratio of HER transitions vs regular transitions
            reward_fn (function): Function to compute reward (achieved_goal, desired_goal) -> reward
        """
        self.device = device
        self.max_size = max_size
        self.her_ratio = her_ratio
        self.reward_fn = reward_fn if reward_fn else self.default_reward_fn
        
        # Separate storage for each component
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32).to(device)
        self.achieved_goals = torch.zeros((max_size, goal_dim), dtype=torch.float32).to(device)
        self.desired_goals = torch.zeros((max_size, goal_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32).to(device)
        
        self.ptr = 0  # Pointer to current position in buffer
        self.size = 0  # Current size of buffer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # Temporary storage for current episode
        self.episode_transitions = []

    def default_reward_fn(self, achieved_goal, desired_goal):
        """Default sparse reward function"""
        # Compute distance between achieved and desired goal
        distance = torch.norm(achieved_goal - desired_goal, dim=-1)
        # Sparse reward: 0 if close enough, -1 otherwise
        return -(distance > 0.05).float()

    def add(self, state, action, next_state, achieved_goal, desired_goal, done):
        """Add a transition to the current episode buffer"""
        transition = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'done': done
        }
        self.episode_transitions.append(transition)

    
    
    #TODO: Make the other one call this store_episode and store the labeled her things 
    # CHECK IF HE LOGIC IS CORRECT modify it, then check with the other classes to match thr psuedocode
    def store_episode(self):
        """Store the current episode in the buffer with HER relabeling"""
        episode_length = len(self.episode_transitions)
        
        # Convert episode data to tensors
        states = torch.stack([t['state'] for t in self.episode_transitions])
        actions = torch.stack([t['action'] for t in self.episode_transitions])
        next_states = torch.stack([t['next_state'] for t in self.episode_transitions])
        achieved_goals = torch.stack([t['achieved_goal'] for t in self.episode_transitions])
        desired_goals = torch.stack([t['desired_goal'] for t in self.episode_transitions])
        dones = torch.tensor([t['done'] for t in self.episode_transitions], dtype=torch.float32).unsqueeze(1)
        
        # Compute original rewards
        rewards = self.reward_fn(achieved_goals, desired_goals)
        
        # Store original transitions
        self._store(
            states, actions, rewards, next_states,
            achieved_goals, desired_goals, dones
        )
        
        # HER relabeling: create additional transitions with new goals
        for t in range(episode_length):
            # Sample future states in the same episode as new goals
            future_offset = np.random.randint(t, episode_length)
            new_goal = self.episode_transitions[future_offset]['achieved_goal']
            
            # Compute new reward with this goal
            new_reward = self.reward_fn(
                self.episode_transitions[t]['achieved_goal'],
                new_goal
            )
            new_done = (new_reward == 0).float()  # Done if reward is 0 (success)
            
            # Store HER transition
            self._store(
                self.episode_transitions[t]['state'],
                self.episode_transitions[t]['action'],
                new_reward,
                self.episode_transitions[t]['next_state'],
                self.episode_transitions[t]['achieved_goal'],
                new_goal,
                new_done
            )
        
        # Clear episode buffer
        self.episode_transitions = []

    def _store(self, states, actions, rewards, next_states, achieved_goals, desired_goals, dones):
        """Internal method to store transitions in the buffer"""
        batch_size = states.shape[0]
        
        # Handle case where we're adding a single transition
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            achieved_goals = achieved_goals.unsqueeze(0)
            desired_goals = desired_goals.unsqueeze(0)
            dones = dones.unsqueeze(0)
        
        # Calculate indices where we'll store these transitions
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        
        # Store transitions
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.achieved_goals[indices] = achieved_goals
        self.desired_goals[indices] = desired_goals
        self.dones[indices] = dones
        
        # Update pointer and size
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        """Sample a batch of transitions, including HER transitions"""
        # Calculate how many HER vs regular transitions to sample
        her_batch_size = int(batch_size * self.her_ratio)
        regular_batch_size = batch_size - her_batch_size
        
        # Sample regular transitions (original goals)
        regular_indices = np.random.randint(0, self.size, size=regular_batch_size)
        
        # Sample HER transitions (relabeled goals)
        her_indices = np.random.randint(0, self.size, size=her_batch_size)
        
        # Combine indices
        indices = np.concatenate([regular_indices, her_indices])
        
        # Get batch data
        batch = {
            'state': self.states[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices],
            'next_state': self.next_states[indices],
            'achieved_goal': self.achieved_goals[indices],
            'desired_goal': self.desired_goals[indices],
            'done': self.dones[indices],
        }
        
        # Concatenate state and goal for policy input
        batch['state_goal'] = torch.cat([batch['state'], batch['desired_goal']], dim=1)
        batch['next_state_goal'] = torch.cat([batch['next_state'], batch['desired_goal']], dim=1)
        
        return batch

    def __len__(self):
        return self.size