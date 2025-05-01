import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from collections import deque
from torchvision.transforms import Compose, ToPILImage, Grayscale, Resize, ToTensor
import gc

# Dueling DQN network copied from train.py (no imports)
class DuelingCNN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super().__init__()
        # convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # compute conv output size for FC layer
        conv_output_size = 64 * 7 * 7  # (84→(84−8)/4+1=20→…→7)
        # shared fully-connected
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU()
        )
        # dueling streams
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: (batch, C, H, W)
        x = self.conv(x)
        x = self.fc(x)
        value = self.value_stream(x)               # (batch,1)
        adv   = self.advantage_stream(x)           # (batch,n_actions)
        # combine into Q-values
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q

class Agent(object):
    """DQN agent with frame-skip and frame-stack for SuperMarioBros."""
    def __init__(self):
        # action space matches COMPLEX_MOVEMENT
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

        # device & model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DuelingCNN(input_channels=4, n_actions=self.action_space.n).to(self.device)
        self.model.eval()

        # load trained weights from model.pth
        try:
            ckpt = torch.load('model.pth', map_location=self.device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
            else:
                self.model.load_state_dict(ckpt)
            print("[Agent] Loaded model.pth successfully.")
        except Exception as e:
            print(f"[Agent] Failed to load model.pth: {e}")
            print("[Agent] Using untrained network.")

        # preprocessing: grayscale + resize to 84×84
        self.transform = Compose([
            ToPILImage(),
            Grayscale(num_output_channels=1),
            Resize((84, 84)),
            ToTensor()  # returns float tensor in [0,1], shape (1,84,84)
        ])

        # frame-stack (k=4)
        self.frame_stack = deque(maxlen=4)
        self.first = True  # flag for first call to act

        # skip-frame: repeat each chosen action for SKIP frames
        SKIP_FRAMES = 4
        self.skip_frames = SKIP_FRAMES - 1  # after selecting one, skip the next SKIP_FRAMES-1
        self.skip_count = 0
        self.last_action = 0

        # for occasional garbage collection
        self.step_counter = 0

    def act(self, observation):
        """
        Args:
            observation (np.ndarray): raw RGB frame, shape (240,256,3), dtype=uint8
        Returns:
            int: chosen action index
        """
        # step counter for GC
        self.step_counter += 1
        if self.step_counter % 50 == 0:
            gc.collect()

        # ensure C-contiguous array
        obs = np.ascontiguousarray(observation)
        # check raw shape
        if obs.shape != (240, 256, 3):
            raise ValueError(f"Expected observation shape (240,256,3), but got {obs.shape}")

        # preprocess: grayscale & resize → (84,84)
        frame = self.transform(obs).squeeze(0).numpy()  # np.ndarray, shape (84,84)

        # initialize frame stack on first call
        if self.first:
            self.frame_stack.clear()
            for _ in range(4):
                self.frame_stack.append(frame)
            self.first = False

        # if skipping frames, return last action
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # otherwise, push new frame and stack
        self.frame_stack.append(frame)
        stacked = np.stack(self.frame_stack, axis=0)  # shape (4,84,84)

        # to tensor, add batch dim: (1,4,84,84)
        state = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        # forward pass
        with torch.no_grad():
            q = self.model(state)  # (1, n_actions)
        action = int(q.argmax(dim=1).item())

        # update skip and last action
        self.last_action = action
        self.skip_count = self.skip_frames

        return action
