import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cpu")
        self.q_net = DuelingCNN(4, self.action_space)
        self.q_net.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.q_net.eval()

    def act(self, observation):
        state_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())