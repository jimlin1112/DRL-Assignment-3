# ===== student_agent.py =====
import os
import gym
import torch
import numpy as np
from collections import deque
from PIL import Image

# Preprocess for agent
def preprocess(frame):
    img = Image.fromarray(frame)
    img = img.convert('L').resize((84,84))
    arr = np.array(img, dtype=np.uint8)
    return arr

class Agent(object):
    """Rainbow DQN Agent"""
    def __init__(self):
        from train import RainbowDQN, V_MIN, V_MAX, ATOM_SIZE
        n_actions = 12
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = RainbowDQN(4, n_actions).to(self.device)
        ckpt = os.path.join(os.path.dirname(__file__), 'models', 'rainbow_mario.pth')
        self.policy_net.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.policy_net.eval()
        self.frames = deque(maxlen=4)

    def act(self, observation):
        frame = np.array(observation)
        s = preprocess(frame)
        while len(self.frames) < 4:
            self.frames.append(s)
        self.frames.append(s)
        state = np.stack(self.frames, axis=0)
        state = torch.from_numpy(state).float().to(self.device) / 255.0
        with torch.no_grad():
            return self.policy_net.act(state)