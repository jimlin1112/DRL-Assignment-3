import gym
import torch
import torch.nn as nn
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Dueling CNN architecture matching train.py
class DuelingCNN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(DuelingCNN, self).__init__()
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
        # Fully connected layer
        conv_output_size = 64 * 7 * 7  # for 84x84 input
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU()
        )
        # Dueling streams: value and advantage
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        value = self.value_stream(x)
        adv = self.advantage_stream(x)
        # Combine value and advantage into Q-values
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q

class Agent(object):
    """Agent that loads a pretrained DQN model and acts deterministically."""
    def __init__(self):
        # Use CPU for evaluation to comply with leaderboard requirements
        self.device = torch.device("cpu")
        # Action space size matches COMPLEX_MOVEMENT
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))
        # Initialize model
        self.model = DuelingCNN(input_channels=4, n_actions=len(COMPLEX_MOVEMENT)).to(self.device)
        # Load pretrained weights from model.pth, prefer weights_only to avoid pickle risks
        try:
            checkpoint = torch.load("model.pth", map_location=self.device, weights_only=True)
        except TypeError:
            # fallback if weights_only not supported in this torch version
            checkpoint = torch.load("model.pth", map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint)
            print("Successfully loaded model.pth")
        except Exception as e:
            print(f"Error loading model.pth: {e}")
        # Set model to evaluation mode
        self.model.eval()

    def act(self, observation):
        # observation: numpy array with shape (4, 84, 84)
        # Ensure positive strides by copying the array
        obs = observation.copy()
        # Convert to torch tensor and add batch dimension
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
            action = int(q_values.argmax(dim=1).item())
        return action
