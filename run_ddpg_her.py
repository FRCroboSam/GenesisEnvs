




class HERReplayBuffer:
    def __init__(self, capacity, state_dim, goal_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, state_dim))
        self.goals = torch.zeros((capacity, goal_dim))
        self.actions = torch.zeros((capacity, action_dim))
        self.rewards = torch.zeros((capacity, 1))
        self.next_states = torch.zeros((capacity, state_dim))
        self.achieved_goals = torch.zeros((capacity, goal_dim))
        self.dones = torch.zeros((capacity, 1))
        self.pos = 0

    def add(self, state, goal, action, reward, next_state, achieved_goal, done):
        idx = self.pos % self.capacity
        self.states[idx] = torch.FloatTensor(state)
        self.goals[idx] = torch.FloatTensor(goal)
        self.actions[idx] = torch.FloatTensor(action)
        self.rewards[idx] = torch.FloatTensor([reward])
        self.next_states[idx] = torch.FloatTensor(next_state)
        self.achieved_goals[idx] = torch.FloatTensor(achieved_goal)
        self.dones[idx] = torch.FloatTensor([done])
        self.pos += 1

    def sample(self, batch_size, relabel_ratio=0.8):
        # Sample random transitions
        idxs = np.random.randint(0, min(self.pos, self.capacity), size=batch_size)
        states = self.states[idxs].to(self.device)
        goals = self.goals[idxs].to(self.device)
        actions = self.actions[idxs].to(self.device)
        rewards = self.rewards[idxs].to(self.device)
        next_states = self.next_states[idxs].to(self.device)
        achieved_goals = self.achieved_goals[idxs].to(self.device)
        dones = self.dones[idxs].to(self.device)

        # HER: Relabel some goals with achieved_goals
        relabel_mask = torch.rand(batch_size) < relabel_ratio
        new_goals = torch.where(relabel_mask.unsqueeze(1), achieved_goals, goals)
        new_rewards = self.compute_reward(achieved_goals, new_goals)  # Custom reward function

        return states, new_goals, actions, new_rewards, next_states, dones