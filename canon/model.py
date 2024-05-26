



"""

model inputs:
obstacle positions
target position
canon position
canon angle
canon propulsion

outputs:
fire (boolean: sign(sigmoid y1)): whether to fire the canon
rotate_amount
propulsion_adjustment

fitness:
- for distance from closest projectile to target
+++ for hitting target
- for number of projectiles fired


"""


import torch
import torch.nn as nn
import torch.optim as optim



class AgentBrain(nn.Module):

    def __init__(self, num_obstacles):
        super().__init__()

        self.linear1 = nn.Linear(2 * num_obstacles + 2 + 2 + 1 + 1, 20)
        self.linear2 = nn.Linear(20, 1 + 1 + 1)
        self.stack = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
        )

    def forward(self, x):
        return self.stack(x)