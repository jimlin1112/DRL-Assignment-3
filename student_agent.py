import gym
import torch
import torch.nn as nn
from collections import deque
from torchvision import transforms as T
import numpy as np
import gc

# Import network architecture and action space
from train import DuelingCNN, COMPLEX_MOVEMENT

class Agent(object):
    """Agent that loads a pretrained DQN model and acts deterministically with frame skipping and stacking."""
    def __init__(self):
        # Use CPU for evaluation to comply with leaderboard requirements
        self.device = torch.device("cpu")
        # Discrete action space matching COMPLEX_MOVEMENT
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))
        # Initialize dueling DQN
        self.model = DuelingCNN(input_channels=4, n_actions=len(COMPLEX_MOVEMENT)).to(self.device)
        # Load pretrained weights safely
        try:
            checkpoint = torch.load("model.pth", map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load("model.pth", map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint)
            print("Successfully loaded model.pth")
        except Exception as e:
            print(f"Error loading model.pth: {e}")
        self.model.eval()

        # Preprocessing: convert to grayscale, resize to 84x84
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.Resize((84, 84)),
            T.ToTensor(),  # outputs shape [1,84,84]
        ])

        # Frame stack for temporal context
        self.frame_stack = deque(maxlen=4)
        self.first = True
        # Frame skipping parameters
        self.skip = 4
        self.skip_count = 0
        self.last_action = 0
        # Step counter for optional garbage collection
        self.step_counter = 0

    def act(self, observation):
        # Ensure array has positive strides
        obs = np.ascontiguousarray(observation)
        # Preprocess: grayscale + resize -> numpy (84,84)
        processed = self.transform(obs).squeeze(0).numpy()

        # Initialize frame stack on first call
        if self.first:
            self.frame_stack.clear()
            for _ in range(4):
                self.frame_stack.append(processed)
            self.first = False

        # Frame skip: return last action if skipping
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # Append new frame and stack
        self.frame_stack.append(processed)
        stacked = np.stack(self.frame_stack, axis=0)  # shape (4,84,84)
        # Convert to tensor and add batch dimension
        state = torch.tensor(stacked, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            q_values = self.model(state)
            action = int(q_values.argmax(dim=1).item())

        # Update skip/frame state
        self.last_action = action
        self.skip_count = self.skip - 1
        # Periodic garbage collection to reduce memory spikes
        self.step_counter += 1
        if self.step_counter % 50 == 0:
            gc.collect()

        return action
