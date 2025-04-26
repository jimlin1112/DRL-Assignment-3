import cv2
import torch
import numpy as np
import torch.nn as nn
from collections import deque
import random

# ------------------------------------------------------------
# Utility: frame preprocessing (same as in training)
# ------------------------------------------------------------
def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB (240x256x3) frame to 84x84 grayscale image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return gray

# ------------------------------------------------------------
# Noisy Linear layer (same implementation as training, noise will be disabled in eval mode)
# ------------------------------------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out', torch.zeros(out_features))
        # Initialization
        bound = 1 / np.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma_init * bound)
        nn.init.constant_(self.bias_sigma, sigma_init * bound)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Sample noise and apply factorized Gaussian noise
            self.noise_in.normal_()
            self.noise_out.normal_()
            eps_in = self.noise_in.sign() * torch.sqrt(self.noise_in.abs())
            eps_out = self.noise_out.sign() * torch.sqrt(self.noise_out.abs())
            weight_noise = torch.outer(eps_out, eps_in)
            bias_noise = eps_out
            weight = self.weight_mu + self.weight_sigma * weight_noise
            bias = self.bias_mu + self.bias_sigma * bias_noise
        else:
            # In eval mode, use mean weights (no noise)
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

# ------------------------------------------------------------
# DQN Network (dueling network with noisy layers, same as training architecture)
# ------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv_output_size = 64 * 7 * 7  # 3136
        self.fc_value = NoisyLinear(self.conv_output_size, 512)
        self.fc_advantage = NoisyLinear(self.conv_output_size, 512)
        self.value_out = NoisyLinear(512, 1)
        self.advantage_out = NoisyLinear(512, num_actions)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_net(x)
        features = features.view(features.size(0), -1)
        value = self.value_out(nn.functional.relu(self.fc_value(features)))
        advantage = self.advantage_out(nn.functional.relu(self.fc_advantage(features)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# Action space for COMPLEX_MOVEMENT (12 discrete actions)
NUM_ACTIONS = 12
action_space = range(NUM_ACTIONS)  # just to sample if needed (replaced by actual env action_space if available)

device = torch.device("cpu")  # ensure CPU mode

# Load the trained model weights
policy_net = DQN(NUM_ACTIONS).to(device)
WEIGHT_PATH = "mario_rainbow.pth"
try:
    state_dict = torch.load(WEIGHT_PATH, map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()  # set network to evaluation mode (no noisy exploration)
    use_network = True
except FileNotFoundError:
    print(f"[student_agent] WARNING: '{WEIGHT_PATH}' not found. Agent will act randomly.")
    use_network = False

# ------------------------------------------------------------
# Agent class for evaluation
# ------------------------------------------------------------
class Agent:
    """Rainbow DQN Agent for Super Mario Bros (COMPLEX_MOVEMENT)."""
    def __init__(self):
        # Frame buffer for stacking last 4 frames
        self.frames = deque(maxlen=4)
        # Variables for action repeat (skip frames)
        self.skip_count = 0
        self.last_action = 0

    def act(self, observation: np.ndarray) -> int:
        """Select an action given the current observation (one frame)."""
        # If no trained network is available, fall back to a random action
        if not use_network:
            return random.choice(action_space)
        # On skipped frames, repeat the last action (no need to recompute)
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action
        # Preprocess the current frame and add to deque
        frame = _preprocess_frame(observation)
        self.frames.append(frame)
        # If this is the start of an episode, pad the deque to 4 frames
        while len(self.frames) < 4:
            self.frames.append(frame)
        # Prepare state tensor (4x84x84) for network
        state = np.stack(self.frames, axis=0)  # shape (4,84,84)
        state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0) / 255.0
        # Compute Q-values and choose the best action
        with torch.no_grad():
            q_values = policy_net(state_tensor)  # shape (1, 12)
            action = int(q_values.argmax(dim=1).item())
        # Set up to skip the next 3 frames with the same action
        self.last_action = action
        self.skip_count = 4 - 1  # already executed this action for 1 frame, 3 more to skip
        return action
