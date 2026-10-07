import torch
import torch.nn as nn
import numpy as np

class ElectronEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1),
            nn.ReLU(),
            nn.Conv3d(16,32,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)

        return self.fc(x)



