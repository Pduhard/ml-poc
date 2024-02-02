import torch
from torch import nn

class RoomPopulationModel(nn.Module):

    def __init__(self, input_size=3408, output_size=1608):
        super(RoomPopulationModel, self).__init__()
        # sequential taking a 2d input of size (9, N) and outputting a 2d output of size (8, N)
        self.seq = nn.Sequential(
            nn.Linear(input_size, input_size * 4, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_size * 4, input_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(input_size, output_size, dtype=torch.float32),
        )
        
    
    def forward(self, x):
        return self.seq(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)