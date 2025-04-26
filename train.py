import gym
import numpy as np
import random, math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros import make as make_mario

# --------------------------------------------------------
# Environment setup (Super Mario Bros with Complex Movement)
# --------------------------------------------------------
env = make_mario('SuperMarioBros-v0')
# Wrap the environment to use COMPLEX_MOVEMENT (12 discrete actions)
try:
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
except ImportError:
    # If gym_super_mario_bros has a different import structure
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    env = gym.wrappers.FrameSkipping(env, COMPLEX_MOVEMENT)  # fallback (adjust as needed)

# Set random seeds for reproducibility (optional)
env.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# --------------------------------------------------------
# Hyperparameters 
# --------------------------------------------------------
NUM_ACTIONS = env.action_space.n        # should be 12 for COMPLEX_MOVEMENT
GAMMA = 0.99                            # discount factor
N_STEPS = 3                             # N-step return length
GAMMA_N = GAMMA ** N_STEPS              # discount factor for N-step returns
LR = 1e-4                               # learning rate for optimizer
TRAINING_STEPS = 200000                 # total training action steps (with frame skip)
INITIAL_EXPLORE_STEPS = 10000           # warm-up steps with random actions (populate replay)
BATCH_SIZE = 32                         # batch size for training
MEMORY_CAPACITY = 50000                 # replay buffer capacity (experience count)
TARGET_UPDATE_FREQ = 1000               # how often to update target network (in steps)
ALPHA = 0.6                             # PER alpha (priority exponent)
BETA_START = 0.4                        # initial PER beta (importance-sampling exponent)
BETA_FRAMES = TRAINING_STEPS            # frames over which to anneal beta to 1.0

# --------------------------------------------------------
# Utility: frame preprocessing (grayscale and downsample)
# --------------------------------------------------------
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB (240x256x3) frame to 84x84 grayscale image (uint8)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize with interpolation (area interpolation for downsampling)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return gray  # returns 84x84 uint8 image

# --------------------------------------------------------
# Noisy Linear layer (for NoisyNets exploration)
# --------------------------------------------------------
class NoisyLinear(nn.Module):
    """Linear layer with factorized noise (NoisyNet)."""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable parameters for base weights and noise scale
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # Buffers for noise values (no grad, updated each forward)
        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out', torch.zeros(out_features))
        # Initialize weight and bias
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma_init * bound)
        nn.init.constant_(self.bias_sigma, sigma_init * bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Sample random noise for each input and output dimension
            self.noise_in.normal_()
            self.noise_out.normal_()
            # Apply factorized noise transformation
            eps_in = self.noise_in.sign() * torch.sqrt(self.noise_in.abs())
            eps_out = self.noise_out.sign() * torch.sqrt(self.noise_out.abs())
            # Create noise matrices for weight and bias
            weight_noise = torch.outer(eps_out, eps_in)  # outer product -> shape (out_features, in_features)
            bias_noise = eps_out
            # Apply noise to weights and biases
            weight = self.weight_mu + self.weight_sigma * weight_noise
            bias = self.bias_mu + self.bias_sigma * bias_noise
        else:
            # No noise in evaluation mode (use mean weights)
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# --------------------------------------------------------
# Dueling DQN Network with NoisyNet (Rainbow Network)
# --------------------------------------------------------
class DQN(nn.Module):
    """Dueling DQN network with NoisyLinear layers."""
    def __init__(self, num_actions: int):
        super().__init__()
        # Convolutional feature extractor (same as Nature DQN, for 84x84 inputs)
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # input: 4x84x84 -> output: 32x20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # output: 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # output: 64x7x7
            nn.ReLU()
        )
        # Compute size of conv output to feed into linear layers
        self.conv_output_size = 64 * 7 * 7  # =3136
        # Dueling architecture: separate value and advantage streams
        # Noisy linear layers for exploration
        self.fc_value = NoisyLinear(self.conv_output_size, 512)
        self.fc_advantage = NoisyLinear(self.conv_output_size, 512)
        self.value_out = NoisyLinear(512, 1)
        self.advantage_out = NoisyLinear(512, num_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through conv layers
        features = self.conv_net(x)               # shape: (batch, 64, 7, 7)
        features = features.view(features.size(0), -1)  # flatten
        # Compute value and advantage streams
        value_hidden = F.relu(self.fc_value(features))
        advantage_hidden = F.relu(self.fc_advantage(features))
        value = self.value_out(value_hidden)            # shape: (batch, 1)
        advantage = self.advantage_out(advantage_hidden) # shape: (batch, num_actions)
        # Combine to get Q values: Q = V + A - mean(A)
        # Expand value to (batch, num_actions) for broadcasting
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# --------------------------------------------------------
# Prioritized Experience Replay Buffer 
# --------------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        # Memory arrays
        self.states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.uint8)
        # Priority array
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0  # initial max priority (new transitions get this by default)
        # Pointers and size
        self.size = 0    # current number of elements
        self.next_idx = 0  # next index to overwrite (if memory full)
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add a new transition to the buffer with priority."""
        idx = self.next_idx
        # Store the transition (state and next_state are 84x84x4 frame stacks as uint8)
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        # Assign priority for new transition
        self.priorities[idx] = self.max_priority
        # Update pointers
        if self.size < self.capacity:
            self.size += 1
        self.next_idx = (self.next_idx + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float):
        """Sample a batch of transitions with indices and importance-sampling weights."""
        if self.size == 0:
            raise ValueError("Replay buffer is empty, cannot sample")
        # Get priorities for all entries and compute probabilities
        probs = self.priorities[:self.size] ** 1.0  # Already ^alpha in priorities, use exponent 1.0 here
        probs /= probs.sum()
        # Sample indices according to probabilities
        indices = np.random.choice(self.size, batch_size, p=probs)
        # Compute importance-sampling weights
        # w_i = (N * P(i))^(-beta) / max_weight
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        # Normalize weights by max weight in batch to scale <= 1
        weights /= weights.max()  
        weights = np.array(weights, dtype=np.float32)
        # Fetch sampled experiences
        batch_states = self.states[indices]
        batch_next_states = self.next_states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_dones = self.dones[indices]
        return indices, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled transitions after learning."""
        for idx, pr in zip(indices, priorities):
            # Set new priority as error^alpha, keep track of max
            self.priorities[idx] = pr
            if pr > self.max_priority:
                self.max_priority = pr

# Initialize the replay buffer
memory = PrioritizedReplayBuffer(MEMORY_CAPACITY, alpha=ALPHA)

# --------------------------------------------------------
# Initialize networks and optimizer
# --------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(NUM_ACTIONS).to(device)
target_net = DQN(NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())  # start with same weights
target_net.eval()  # target network in inference mode
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

# Variables for PER beta schedule
beta = BETA_START
beta_increment = (1.0 - BETA_START) / BETA_FRAMES

# --------------------------------------------------------
# Training loop (with N-step returns and Double DQN updates)
# --------------------------------------------------------
n_step_buffer = deque(maxlen=N_STEPS)  # buffer for N-step transition tracking
episode_rewards = []
episode = 0

state = env.reset()
# Preprocess initial observation and initialize frame stack
frame = preprocess_frame(state)  # 84x84
frame_stack = deque([frame] * 4, maxlen=4)  # start with 4 identical frames
state_stack = np.array(frame_stack, dtype=np.uint8)  # shape (4,84,84)

print("Starting training...")
for t in range(1, TRAINING_STEPS + 1):
    # Select an action (NoisyNet provides exploration, so we take argmax Q)
    state_tensor = torch.as_tensor(state_stack, device=device, dtype=torch.float32).unsqueeze(0) / 255.0
    with torch.no_grad():
        q_values = policy_net(state_tensor)        # shape (1, num_actions)
        action = int(q_values.argmax(dim=1).item()) 
    # During initial exploration, override action with random to fill replay
    if t <= INITIAL_EXPLORE_STEPS:
        action = env.action_space.sample()
    
    # Execute action in environment with skip (repeat action for 4 frames)
    total_reward = 0.0
    done = False
    for _ in range(4):  # skip 4 frames
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    episode_rewards.append(total_reward)
    next_frame = preprocess_frame(next_obs)  # preprocess new frame
    # Update frame stack for next state
    frame_stack.append(next_frame)
    next_state_stack = np.array(frame_stack, dtype=np.uint8)
    # Save one-step transition to N-step buffer
    n_step_buffer.append((state_stack, action, total_reward, next_state_stack, done))
    # If we have N-step transition, add to replay memory
    if len(n_step_buffer) == N_STEPS:
        # Compute N-step cumulative reward and get nth next state
        R = 0.0
        for idx, (_, _, r, _, _) in enumerate(n_step_buffer):
            R += (GAMMA ** idx) * n_step_buffer[idx][2]  # accumulate discounted rewards
        _, _, _, next_state_n, done_n = n_step_buffer[-1]
        state_n, action_n, _, _, _ = n_step_buffer[0]
        memory.add(state_n, action_n, R, next_state_n, done_n)
    # If episode ends before accumulating N steps, we'll flush the buffer later
    
    # Training step (update network) if conditions are met
    if t > INITIAL_EXPLORE_STEPS and memory.size >= BATCH_SIZE:
        # Sample a batch from replay memory with PER
        indices, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weights = memory.sample(BATCH_SIZE, beta)
        beta = min(1.0, beta + beta_increment)  # anneal beta towards 1
        
        # Convert to torch tensors
        batch_states = torch.from_numpy(batch_states).to(device).float() / 255.0      # shape (B,4,84,84)
        batch_next_states = torch.from_numpy(batch_next_states).to(device).float() / 255.0
        batch_actions = torch.from_numpy(batch_actions).to(device).long()
        batch_rewards = torch.from_numpy(batch_rewards).to(device).float()
        batch_dones = torch.from_numpy(batch_dones).to(device).float()
        weights_tensor = torch.from_numpy(weights).to(device).float()
        
        # Double DQN target calculation
        with torch.no_grad():
            # Online network selects best action for next states
            next_q_values = policy_net(batch_next_states)
            best_actions = next_q_values.argmax(dim=1, keepdim=True)        # shape (B,1)
            # Target network evaluates those actions
            target_q_values = target_net(batch_next_states).gather(1, best_actions).squeeze(1)  # shape (B,)
            # Compute target: N-step discounted reward + gamma^N * Q_target(next) (if not done)
            target = batch_rewards + (GAMMA_N * target_q_values * (1 - batch_dones))
        # Current Q estimates for taken actions
        current_q = policy_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        # Huber loss (smooth L1) weighted by PER importance-sampling weights
        loss = F.smooth_l1_loss(current_q, target, reduction='none')
        loss = (loss * weights_tensor).mean()
        
        # Backpropagate and update network
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional, to avoid out-of-range gradients)
        nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        optimizer.step()
        
        # Update priorities in replay buffer based on new TD errors
        td_errors = (target - current_q).detach().cpu().numpy()
        new_priorities = (np.abs(td_errors) + 1e-3) ** ALPHA
        memory.update_priorities(indices, new_priorities)
        
        # Periodically update the target network (hard update)
        if t % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    # Prepare for next step
    state_stack = next_state_stack
    # If episode ended, flush remaining N-step buffer and reset environment
    if done:
        # Flush n-step buffer for transitions shorter than N
        while n_step_buffer:
            # Compute remaining n-step reward for buffer length L < N
            L = len(n_step_buffer)
            R = 0.0
            for idx, (_, _, r, _, _) in enumerate(n_step_buffer):
                R += (GAMMA ** idx) * n_step_buffer[idx][2]
            state_0, action_0, _, next_state_L, done_L = n_step_buffer[0]
            # Note: done_L is True here (episode ended)
            memory.add(state_0, action_0, R, next_state_L, done_L)
            n_step_buffer.popleft()
        # Reset environment for next episode
        state = env.reset()
        # Reset frame stack with the first frame of new episode
        frame = preprocess_frame(state)
        frame_stack = deque([frame] * 4, maxlen=4)
        state_stack = np.array(frame_stack, dtype=np.uint8)
        episode += 1
        # (Optional) Logging: print(f"Episode {episode} finished with reward {sum(episode_rewards)}")
        print(f"Episode {episode} finished with reward {sum(episode_rewards)}, Step: {t}")
        episode_rewards.clear()

# --------------------------------------------------------
# Save the trained model 
# --------------------------------------------------------
torch.save(policy_net.state_dict(), "mario_rainbow.pth")
print("Training completed, model saved as mario_rainbow.pth")
