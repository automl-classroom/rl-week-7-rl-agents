"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed


class RNDModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int, output_dim: int, n_layers: int = 2
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )

        self.seed = seed
        obs_dim = env.observation_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rnd_target = RNDModel(
            obs_dim, rnd_hidden_size, rnd_hidden_size, rnd_n_layers
        ).to(self.device)
        self.rnd_predictor = RNDModel(
            obs_dim, rnd_hidden_size, rnd_hidden_size, rnd_n_layers
        ).to(self.device)
        self.rnd_target.eval()  # Target-Netzwerk bleibt fix

        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=rnd_lr)
        self.rnd_reward_weight = rnd_reward_weight
        self.rnd_update_freq = rnd_update_freq
        # TODO: initialize the RND networks

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # TODO: get states and next_states from the batch
        # TODO: compute the MSE
        # TODO: update the RND network
        states = np.array([transition[0] for transition in training_batch])
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            target_embeddings = self.rnd_target(states_tensor)

        predicted_embeddings = self.rnd_predictor(states_tensor)
        mse_loss = torch.mean((predicted_embeddings - target_embeddings) ** 2)

        self.rnd_optimizer.zero_grad()
        mse_loss.backward()
        self.rnd_optimizer.step()

        return mse_loss.item()

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        # TODO: predict embeddings
        # TODO: get error
        self.rnd_predictor.eval()
        with torch.no_grad():
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            )
            target_embedding = self.rnd_target(state_tensor)
            predictor_embedding = self.rnd_predictor(state_tensor)
            mse = torch.mean((predictor_embedding - target_embedding) ** 2)
            return mse.item()

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # TODO: apply RND bonus
            bonus = self.get_rnd_bonus(state)
            reward += self.rnd_reward_weight * bonus

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if frame % self.rnd_update_freq == 0:
                self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards)
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        # Saving to .csv for simplicity
        # Could also be e.g. npz
        print("Training complete.")
        training_data = pd.DataFrame({"steps": steps, "rewards": episode_rewards})
        training_data.to_csv(f"training_data_seed_{self.seed}.csv", index=False)


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = RNDDQNAgent(
        env,
        seed=cfg.seed,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        # RND-Parameter
        rnd_hidden_size=128,
        rnd_lr=1e-3,
        rnd_update_freq=1000,
        rnd_n_layers=2,
        rnd_reward_weight=0.1,
    )

    agent.train(num_frames=cfg.train.num_frames, eval_interval=cfg.train.eval_interval)


if __name__ == "__main__":
    main()
