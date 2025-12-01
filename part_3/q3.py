import random
import copy
from collections import deque

import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import matplotlib.pyplot as plt

import utils
from n_network import DQN


env = gym.make("CartPole-v1", render_mode="rgb_array")

    
class PoleAgent:
    def __init__(
        self,
        q_network: nn.Module = None,
        replay_size: int = 10000,
        gamma: float = 0.99,
        epsilon_start: float = 1,
        min_epsilon: float = 0.05,
        eps_decay_factor: float = 0.99,
        optimizer_name: str = "rmsprop", 
        lr: float = 1e-3,
        loss_type: str = "mse",# "huber" | "mse"
        device: str = "cpu",    
        action_shape: int = 2,
        state_shape: int = 4,
        n_episodes:int = 100,
        batch_size: int = 16,
        target_update_freq: int = 10,
        min_reply_size: int = 160,
        TAU=0.005,
        env=None
    ):
        self.device = device
        if q_network is None:
                raise ValueError("No network was passed")
        self.q_network = q_network.to(device)
        self.target_network = copy.deepcopy(q_network).to(self.device)
        self.target_network.eval() 

        if not env:
                raise ValueError("No env was passed")
        self.env = env

        self.experience_replay = deque(maxlen=replay_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.min_epsilon = min_epsilon
        self.eps_decay_factor = eps_decay_factor
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_reply_size = min_reply_size
        self.train_steps_done = 0
        self.eps_threshold = 0
        self.TAU = TAU

        if loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        
        self.optimizer = self._build_optimizer(
            optimizer_name=optimizer_name,
            lr=lr,
        )
    
    def _build_optimizer(self, optimizer_name: str, lr: float):
        """Factory for different optimizers."""
        name = optimizer_name.lower()

        if name == "sgd":
            return optim.SGD(self.q_network.parameters(), lr=lr, momentum=0.9)
        elif name == "rmsprop":
            return optim.RMSprop(self.q_network.parameters(), lr=lr, alpha=0.95, eps=1e-8)
        elif name == "adam":
            return optim.Adam(self.q_network.parameters(), lr=lr, amsgrad=True)
        else:
            raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    def _update_epsilon(self):
        """epsilon decay: eps <- max(eps_min, eps * decay_factor)."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_decay_factor)

    def sample_batch(self, batch_size:int):
        """
        Returns tensors on self.device:
            states      : (B, state_dim)   float32
            actions     : (B, 1)           long
            rewards     : (B, 1)           float32
            next_states : (B, state_dim)   float32
            dones       : (B, 1)           float32 (0.0 or 1.0)
        """
        if len(self.experience_replay) < batch_size:
            raise ValueError(
                f"Not enough samples in replay buffer "
                f"({len(self.experience_replay)} < {batch_size})"
            )
        batch =  random.sample(self.experience_replay, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(-1)

        return states, actions, rewards, next_states, dones

    def update_target_network(self):
        """
        Instead of updating the target network every C step we instead add small updates 
        """
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                policy_net_state_dict[key] * self.TAU
                + target_net_state_dict[key] * (1 - self.TAU)
            )
        self.target_network.load_state_dict(target_net_state_dict)

    def add_to_reply(self, state, action, reward, next_state, done): 
        """
        Adding reply to the reply buffer
        """
        self.experience_replay.append(
            (np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done))
        )

    def sample_action(self, state):
        """
        Samples an action based on epsilon-greedy mechaisehm:
        random.sample <= epsilon -> sample random action - for exploration
        random.sample > epsilon -> use the network to compute best q values
        """

        if np.random.rand() <= self.epsilon:
            return np.random.randint(low=0, high=2, size=(1,), dtype=np.int64)[0]
        else:
            state_as_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) 
            with torch.no_grad():
                q_values = self.q_network(state_as_tensor)
                action_idx = q_values.argmax(dim=1).item()
            return action_idx
                        
    def compute_step(self):
        """
        Perform one DDQN update step using a minibatch.
        """ 
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size=self.batch_size)

        q_values = self.q_network(states)
        q_taken = q_values.gather(1, actions)
        with torch.no_grad():
            # 1. Use online network to select action
            q_next_online = self.q_network(next_states)
            next_actions = q_next_online.argmax(dim=1, keepdim=True)

            # 2. Use target network to evaluate that action
            q_next_target = self.target_network(next_states)
            q_next_target_chosen = q_next_target.gather(1, next_actions)

            # 3. Build target (shape [batch,1], like before)
            targets = rewards + self.gamma * q_next_target_chosen * (1.0 - dones)

        loss = self.criterion(q_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100.0)
        self.optimizer.step()

        self.train_steps_done += 1

        # if self.train_steps_done % self.target_update_freq == 0:
        self.update_target_network()

        return loss.item()
    
    def train_agent(self, log_every=100):
        """
        The training loop, iterates for n_episodes.
        logging every K steps, default to 100. Returns the reward and losses.
        """
        losses, rewards = [], []

        for episode in range(self.n_episodes):
            state, _ = self.env.reset()
            done = False
            total_rewards,step = 0,0 
            ep_loss, step_loss = [], []
            while not done:
                action = self.sample_action(state=state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # add to D
                self.add_to_reply(state=state, action=action, next_state=next_state, reward=reward, done=done)
                # s_t+1 <- s_t
                state = next_state
                total_rewards += reward
                if len(self.experience_replay) >= self.min_reply_size:
                    loss = self.compute_step()
                    ep_loss.append(loss)
                    step_loss.append(loss)
            rewards.append(total_rewards)
            mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
            losses.append(mean_loss)   
            self._update_epsilon()

            if (episode + 1) % log_every == 0: 
                recent_rewards = rewards[-log_every:]
                recent_losses = losses[-log_every:]
                avg_reward = float(np.mean(recent_rewards))
                avg_loss = float(np.mean(recent_losses))

                print(
                    f"Episode {episode+1:4d}/{self.n_episodes} | "
                    f"Epsilon = {self.epsilon} | "
                    f"avg_reward({log_every}) = {avg_reward:7.2f} | "
                    f"avg_loss({log_every}) = {avg_loss:8.5f} | "
                    f"buffer_size = {len(self.experience_replay):6d}"
                )

        return rewards, losses,step_loss


def test_agent(agent, env, n_episodes=30):
    for e in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()
            with torch.no_grad():
                action = agent.q_network(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        print(f"Episode {e + 1}: Episode Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    seed = 42
    utils.set_seed(seed)
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0] 
    n_actions = env.action_space.n             
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dims = [128,128,128]

    q_net = DQN(state_dim=state_dim, action_dim=n_actions, hidden_dims=hidden_dims)

    agent = PoleAgent(
        env=env,
        q_network=q_net,
        action_shape=n_actions,
        state_shape=state_dim,  
        replay_size=1500,
        gamma=0.99,
        epsilon_start=0.9,
        min_epsilon=0.01,
        eps_decay_factor=0.95,
        optimizer_name="Adam",
        lr=5e-4,
        loss_type="huber",
        device=device,
        n_episodes=500,
        min_reply_size=500,
        batch_size=256,
        TAU=0.005
    )

    print(f"Training on device: {device}")
    print(f"Network hidden_dims = {hidden_dims}")

    rewards, losses, step_loss = agent.train_agent(log_every=50)

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    # Plot loss per training step
    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss per Training Step')
    plt.show()
    print("Training finished.")
    print(f"Final average reward (last 20 episodes): {np.mean(rewards[-20:]):.2f}")

    test_env = gym.make('CartPole-v1', render_mode='human')
    test_agent(agent, test_env)



