import time
import gym
import numpy as np
import random
import torch
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
pre_load = True

# Replay buffer stores image frames (uint8) and other transitions.
class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape  # e.g. (4,84,84)
        self.ptr = 0
        self.size = 0
        # Pre-allocate buffers for efficiency
        self.states      = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.dones[self.ptr]       = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size

# CNN with dueling architecture for image inputs.
class DuelingCNN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super().__init__()
        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute size of conv output (should be 64*7*7 = 3136 for 84x84 input)
        conv_output_size = 64 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU()
        )
        # Dueling streams
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        value = self.value_stream(x)
        adv = self.advantage_stream(x)
        # Combine into Q values
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward, done = 0.0, False
        info = {}
        for _ in range(self._skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, info

class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        from torchvision.transforms import Compose, ToPILImage, Grayscale, Resize, ToTensor
        self.transform = Compose([
            ToPILImage(),
            Grayscale(num_output_channels=1),
            Resize((84, 84)),
            ToTensor()
        ])
        # after transform we'll get shape (1,84,84); but we squeeze to (84,84)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(84, 84),
            dtype=np.float32
        )
    def observation(self, state):
        t = self.transform(state)       # torch.Tensor, shape [1,84,84]
        return t.squeeze(0).numpy()   # -> np.ndarray [84,84]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        from collections import deque
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape  # here (84,84)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(shp[0]*k, shp[1], shp[2]) if len(shp)==3 else (k, *shp),
            dtype=np.float32
        )
    def reset(self):
        state = self.env.reset()
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(state)
        return np.stack(self.frames, axis=0)
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.frames.append(state)
        return np.stack(self.frames, axis=0), reward, done, info

class DQNVariant:
    def __init__(self, state_shape, action_size, buffer_capacity=10000,
        batch_size=128, gamma=0.9, lr=0.00025, tau=5e-3, target_update_interval=10000):
        
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Q-networks
        c = state_shape[0]
        self.q_net = DuelingCNN(c, action_size).to(self.device)
        self.target_net = DuelingCNN(c, action_size).to(self.device)
        if pre_load:
            self.q_net.load_state_dict(torch.load(save_path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = F.smooth_l1_loss
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity, state_shape)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_interval = target_update_interval
        self.learn_steps = 0
        # Epsilon-greedy params
        self.eps = 0.01
        self.eps_min = 0.01
        self.eps_decay = 0.9999

    def get_action(self, state: np.ndarray, deterministic=True) -> int:
        # state: np.array of shape (4,84,84)
        if (not deterministic) and (random.random() < self.eps):
            return random.randrange(self.action_size)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def update_target(self, hard=True):
        if hard:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            # Soft update: τ*θ + (1-τ)*θ_target
            for tgt_p, src_p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tgt_p.data.copy_(tgt_p.data * (1.0 - self.tau) + src_p.data * self.tau)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        # Sample a batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute current Q and target Q
        current_q = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network and epsilon
        self.learn_steps += 1
        if self.learn_steps % self.update_interval == 0:
            self.update_target(hard=True)
        # else:
        #     self.update_target(hard=False)
        # Decay epsilon
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


file = open("output.txt", "w")
save_path = "model.pth"

fin = open("record.txt", "r")
for line in fin.readlines():
    file.write(line)
fin.close()


# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)



# Create and wrap environment
env = make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleResize(env)
env = FrameStack(env, 4)
env.seed(SEED)

state_shape = env.observation_space.shape  # should be (4,84,84)
action_size = env.action_space.n

agent = DQNVariant(state_shape, action_size)

total_steps = 0
episode = 2473
start_frame = 1999276
episode_reward = 0
start_time = time.time()
state = env.reset()

pbar = tqdm(range(start_frame+1, int(1e7)+start_frame +1))
for total_steps in pbar:
    action = agent.get_action(state, deterministic=False)
    next_state, reward, done, info = env.step(action)
    agent.memory.add(state, action, reward, next_state, done)
    if total_steps % 4 == 0:
        agent.train()
    state = next_state
    episode_reward += reward
    if done:
        episode += 1
        file.write(f"Episode {episode}\tStep {total_steps}\tTime {time.time() - start_time:.2f}\tEpsilon {agent.eps:.2f}\tReward {episode_reward:.1f}\n")
        file.flush()
        episode_reward = 0
        state = env.reset()
    if total_steps % 10000 == 0:
        torch.save(agent.q_net.state_dict(), save_path)


# Save final model
torch.save(agent.q_net.state_dict(), save_path)
env.close()
file.close()
