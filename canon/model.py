



"""

model inputs:
obstacle positions
target position
canon position
canon angle
canon propulsion

outputs:
fire (boolean: sign(sigmoid y1)): whether to fire the canon
rotate (boolean)
rotate_amount
adjust propulsion(boolean)
propulsion_adjustment

fitness:
- for distance from closest projectile to target
+++ for hitting target
- for number of projectiles fired


"""


import torch
import torch.nn as nn



class AgentBrain(nn.Module):

    def __init__(self, num_obstacles, mutation_rate):
        super().__init__()

        self.num_obstacles = num_obstacles
        self.linear1 = nn.Linear(2 * num_obstacles + 2 + 2 + 1 + 1, 20)
        self.linear2 = nn.Linear(20, 1 + 1 + 1 + 1 + 1)
        self.stack = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.Tanh(),
        )
        self.mutation_rate = mutation_rate

    def forward(self, x):
        return self.stack(x)
    
    def save(self, filename):
        torch.save(self, filename)

    def crossover(self, other):
        child1, child2 = AgentBrain(self.num_obstacles, self.mutation_rate), AgentBrain(self.num_obstacles, self.mutation_rate)
        # child1 = AgentBrain(self.num_obstacles, self.mutation_rate)

        l1_half_size = self.linear1.weight.data.size()[0] // 2
        l2_half_size = self.linear2.weight.data.size()[0] // 2

        # child1_l1 = torch.rand(1).item()
        # child1_l2 = torch.rand(1).item()
        # child2_l1 = torch.rand(1).item()
        # child2_l2 = torch.rand(1).item()

        # child1.linear1.weight.data = child1_l1 * self.linear1.weight.data + (1 - child1_l1) * other.linear1.weight.data
        # child1.linear2.weight.data = child1_l2 * self.linear2.weight.data + (1 - child1_l2) * other.linear2.weight.data
        # child2.linear1.weight.data = child2_l1 * self.linear1.weight.data + (1 - child2_l1) * other.linear1.weight.data
        # child2.linear2.weight.data = child2_l2 * self.linear2.weight.data + (1 - child2_l2) * other.linear2.weight.data

        child1.linear1.weight.data = torch.cat(
            (
                self.linear1.weight.data[:l1_half_size],
                other.linear1.weight.data[l1_half_size:]
            ),
            dim=0,
        )

        child1.linear2.weight.data = torch.cat(
            (
                self.linear2.weight.data[:l2_half_size],
                other.linear2.weight.data[l2_half_size:]
            ),
            dim=0,
        )

        child2.linear1.weight.data = torch.cat(
            (
                self.linear1.weight.data[l1_half_size:],
                other.linear1.weight.data[:l1_half_size]
            ),
            dim=0,
        )

        child2.linear2.weight.data = torch.cat(
            (
                self.linear2.weight.data[l2_half_size:],
                other.linear2.weight.data[:l2_half_size]
            ),
            dim=0,
        )


        child1.mutate()
        child2.mutate()

        return child1, child2
        # return child1

    def mutate(self):
        for param in self.parameters():
            if torch.rand(1).item() < self.mutation_rate:
                param.data += torch.rand_like(param.data) * 0.1

    def decide(self, x):
        return self(torch.Tensor(x))