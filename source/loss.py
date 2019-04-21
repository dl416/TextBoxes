import torch
import torch.nn as nn

class textboxesLoss(nn.Module):
    def __init__(self, alpha=1):
        super(textboxesLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, predict, lines):
        pass


