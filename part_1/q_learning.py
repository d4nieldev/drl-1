import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from tensorboardX import SummaryWriter


def q_learning(
        env: gym.Env,
        tensorboard_writer: SummaryWriter,
        max_episodes: int = 5000,
        max_steps_per_episode: int = 100,
        return_q_table_at: list[int] = [500, 2000],
        learning_rate: float = 0.9, 
        discount_factor: float = 0.9, 
        epsilon: float = 0.5,
        epsilon_min: float = 0.1,
        decay_rate: float = 0.999,
    ) -> list[np.ndarray]:
    """
    Q-Learning algorithm to learn the Q-function for a given environment.
    
    Args:
        env (gym.Env): The environment to train on.
        tensorboard_writer (SummaryWriter): TensorBoard writer for logging.
        max_episodes (int): Maximum number of episodes to train.
        max_steps_per_episode (int): Maximum steps per episode.
        return_q_table_at (list[int]): List of episodes at which to return snapshots of the Q-table.
        learning_rate (float): Learning rate for Q-value updates.
        discount_factor (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate for epsilon-greedy policy.
        epsilon_min (float): Minimum exploration rate for epsilon-greedy policy.
        decay_rate (float): Decay rate for epsilon.
    
    Returns:
        list[np.ndarray]: Snapshots of the Q-table at specified episodes with the final Q-table being the last element in the list.
    """

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    # initialize the table with zeros
    lookup_table = np.zeros((num_states, num_actions), dtype=np.float64)
    q_table_snapshots = []
    total_steps_to_goal = 0
    for episode in tqdm(range(max_episodes), desc="Q-Learning Episodes"):
        s, info = env.reset()
        episode_steps_to_goal = 100
        episode_reward = 0.0
        epsilon = max(epsilon_min, epsilon * decay_rate)
        for step in range(max_steps_per_episode):
            # choose action with decaying epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                a = env.action_space.sample()  # explore
            else:
                # break ties randomly
                max_ids = np.where(lookup_table[s, :] == np.max(lookup_table[s, :]))[0]
                a = np.random.choice(max_ids)  # exploit
            
            # interact with the environment
            next_s, reward, terminated, truncated, info = env.step(a)
            reward = np.float64(reward)
            episode_reward += reward

            # update the Q-function lookup table
            lookup_table[s][a] += learning_rate * (reward + discount_factor * np.max(lookup_table[next_s]) - lookup_table[s][a])

            # update state
            s = next_s

            if terminated or truncated:
                if reward > 0:
                    # reached goal
                    episode_steps_to_goal = step + 1
                break
        
        # reward per episode
        tensorboard_writer.add_scalar('Total Reward', episode_reward, episode)
        
        # avg steps to goal per 100 episodes
        total_steps_to_goal += episode_steps_to_goal
        if (episode + 1) % 100 == 0:
            avg_steps = total_steps_to_goal / 100
            tensorboard_writer.add_scalar('Avg Steps to Goal (per 100 eps)', avg_steps, episode)
            total_steps_to_goal = 0

        # save Q-table snapshots at specified episodes
        if episode + 1 in return_q_table_at:
            q_table_snapshots.append(lookup_table.copy())
    
    return q_table_snapshots + [lookup_table]


def plot_q_table_heatmaps(q_tables: list[np.ndarray], titles: list[str]):
    num_tables = len(q_tables)
    fig, axes = plt.subplots(1, num_tables, figsize=(10 * num_tables, 8))

    for idx, q_table in enumerate(q_tables):
        ax = axes[idx] if num_tables > 1 else axes
        cax = ax.matshow(q_table, cmap='viridis', aspect='auto')
        ax.set_title(titles[idx])
        fig.colorbar(cax, ax=ax)

        ax.set_xlabel('Action', fontsize=12)
        ax.set_ylabel('State', fontsize=12)

        rows, cols = q_table.shape
        for i in range(rows):
            for j in range(cols):
                text_val = f"{q_table[i, j]:.2f}" # Format to 2 decimal places
                
                # Use ax.text to place the number
                # j is x-axis (column), i is y-axis (row)
                ax.text(j, i, text_val, 
                        ha="center", va="center", color="coral")

    plt.tight_layout()
    plt.show()



def main():
    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="4x4",
        is_slippery=False,
        # success_rate=1.0/3.0,
        # reward_schedule=(1, 0, 0)
    )

    log_dir = "rl_logs/frozenlake_v1_" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    q_function_snapshots = q_learning(
        env=env,
        tensorboard_writer=writer,
        max_episodes=5000,
        max_steps_per_episode=100,
        return_q_table_at=[500, 2000],
        learning_rate=0.9,
        discount_factor=0.95,
        epsilon=0.1,
        epsilon_min=0.01,
        decay_rate=0.99,
    )
    writer.close()

    # plot q functions as heatmaps side by side
    plot_q_table_heatmaps(
        q_tables=q_function_snapshots,
        titles=[
            "Q-Table at 500 Steps",
            "Q-Table at 2000 Steps",
            "Final Q-Table"
        ]
    )

    print("Data saved to TensorBoard logs. To visualize, run:\n")
    print(f"tensorboard --logdir={log_dir}\n")


if __name__ == "__main__":
    main()