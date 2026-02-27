import os.path
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch.distributions import Categorical

class MemoryBuffer:
    '''Simple buffer to collect experiences and clear after each update.'''
    def __init__(self, device):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []
        self.device = device

    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]
        
    def get_ordered_trajectories(self, n_agents=None):
        ordered_actions = torch.FloatTensor().to(self.device)
        ordered_states = torch.FloatTensor().to(self.device)
        ordered_logprobs = torch.FloatTensor().to(self.device)
        ordered_rewards = []
        ordered_dones = []
        
        actions = torch.stack(self.actions).to(self.device)
        states = torch.stack(self.states).to(self.device)
        logprobs = torch.stack(self.logprobs).to(self.device)
        
        self.ordered_actions = torch.FloatTensor().to(self.device)

        for index in range(actions.shape[1]):
            if n_agents != None and n_agents == index+1:
                break
            # ordered_states = torch.cat((ordered_states, states[:, index]), 0).to(self.device)
            # ordered_actions = torch.cat((ordered_actions, actions[:, index]), 0).to(self.device)
            # ordered_logprobs = torch.cat((ordered_logprobs, logprobs[:, index]), 0).to(self.device)
            # ordered_rewards.extend(np.asarray(self.rewards)[:, index])
            # ordered_dones.extend(np.asarray(self.dones)[:, index])
            ordered_states = states
            ordered_actions = actions
            ordered_logprobs = logprobs
            ordered_rewards.extend(np.asarray(self.rewards))
            ordered_dones.extend(np.asarray(self.dones))
        return ordered_states, ordered_actions, ordered_logprobs, ordered_rewards, ordered_dones
        
class ActorCritic(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        # self.actor_fc3 = nn.Linear(hidden_size, hidden_size) # Added an extra hidden layer
        # self.actor_fc4 = nn.Linear(hidden_size, hidden_size) # Added another hidden layer
        self.actor_out = nn.Linear(hidden_size, action_size)
        
        # Critic network
        self.critic_fc1 = nn.Linear(state_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        # self.critic_fc3 = nn.Linear(hidden_size, hidden_size) # Added an extra hidden layer
        # self.critic_fc4 = nn.Linear(hidden_size, hidden_size) # Added another hidden layer
        self.critic_out = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        # Actor
        x = self.actor_fc1(state)
        x = f.relu(x)
        x = self.actor_fc2(x)
        x = f.relu(x)
        # x = self.actor_fc3(x)
        # x = f.relu(x)
        # x = self.actor_fc4(x)
        # x = f.relu(x)
        x = self.actor_out(x)
        probs = f.softmax(x, dim=-1)
        
        # Critic
        v = self.critic_fc1(state)
        v = f.relu(v)
        v = self.critic_fc2(v)
        v = f.relu(v)
        # v = self.critic_fc3(v)
        # v = f.relu(v)
        state_value = self.critic_out(v)
        
        return probs, state_value
    
    def act(self, state):
        '''Choose action according to the policy.'''
        probs, _ = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()
    
    def evaluate(self, states, actions):
        probs, state_values = self.forward(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, torch.squeeze(state_values), entropy
        
    
class PPO(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, lr, gamma, device, epsilon_clip=0.2, K_epoch=4):
        super(PPO, self).__init__()
        self.policy = ActorCritic(state_size, hidden_size, action_size).to(device)
        self.policy_old = ActorCritic(state_size, hidden_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

        self.gamma = gamma
        self.device = device
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epoch
        
        self.policy_old.load_state_dict(self.policy.state_dict())

    def take_action(self, states):
        #param states: nparray, size(state_dim,) 
        states_np = np.array(states)[np.newaxis, :]  # (state_dim,) -> (1, state_dim)
        states_tensor = torch.tensor(states_np, dtype=torch.float32).to(self.device) #chuyển trạng thái sang tensor
        action, log_prob = self.policy.act(states_tensor)

        return action, log_prob
    
    def update(self, memory):
        states, actions, log_probs, rewards, dones = memory.get_ordered_trajectories()
        
        discounted_rewards = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            if dones[i] == True:
                discounted_reward = 0
            discounted_reward = rewards[i] + self.gamma*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
            
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        states = states.detach()
        actions = actions.detach()
        old_log_probs = log_probs.detach()
        
        for epoch in range(self.K_epochs):
            new_log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            
            new_log_probs = new_log_probs.squeeze()
            advantages = discounted_rewards - state_values.detach().squeeze()
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            ratios_clipped = torch.clamp(ratios, min=1-self.epsilon_clip, max=1+self.epsilon_clip)
            loss = -torch.min(ratios*advantages, ratios_clipped*advantages) + 0.5*self.MseLoss(state_values, discounted_rewards) - 0.01*dist_entropy
            # loss = self.MseLoss(state_values, discounted_rewards)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            loss_value = loss.mean().item()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss_value

    def save(self, save_dir, epoch_i):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(save_dir, "actor", 'policy_weight_' + str(epoch_i) + '.pth'))
        
        # torch.save({
        #     'model_state_dict': self.critic.state_dict(),
        #     'optimizer_state_dict': self.critic_optimizer.state_dict()
        # }, os.path.join(save_dir, "critic", 'critic_weights_' + str(epoch_i) + '.pth'))

    # def load(self, actor_path, critic_path):
    #     if actor_path and os.path.exists(actor_path):
    #         checkpoint = torch.load(actor_path)
    #         self.actor.load_state_dict(checkpoint['model_state_dict'])
    #         self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #     if critic_path and os.path.exists(critic_path):
    #         checkpoint = torch.load(critic_path)
    #         self.critic.load_state_dict(checkpoint['model_state_dict'])
    #         self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])