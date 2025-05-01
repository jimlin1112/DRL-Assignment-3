import gym
from train import DuelingCNN
import torch
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cpu")
        self.q_net = DuelingCNN(4, COMPLEX_MOVEMENT)
        self.q_net.load_state_dict(torch.load("model.pth"))
        self.q_net.eval()

    def act(self, observation):
        state_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())