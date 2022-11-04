import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BasicBlock']


class BasicBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        config = {
            'in_features': None,
            'out_features': None,
            'init': True
        }
        for k, v in config.items():
            setattr(self, k, kwargs.get(k, v))

        self.linear = nn.Linear(self.in_features, self.out_features)
        if self.init:
            nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        out = F.relu(self.linear(x))
        return out