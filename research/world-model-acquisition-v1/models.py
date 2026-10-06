"""Same two categorical parameterizations as the published pilot."""
import numpy as np
import torch
from torch import nn

def features(states):
    sf = np.zeros((30, 12), dtype=np.float32)
    for i, (x, y, k) in enumerate(states):
        sf[i, x] = sf[i, 5+y] = sf[i, 10+k] = 1
    sa = np.concatenate((np.repeat(sf, 4, axis=0), np.tile(np.eye(4, dtype=np.float32), (30, 1))), axis=1)
    return torch.from_numpy(sf), torch.from_numpy(sa)

class Direct(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 30))
    def forward(self, sa, candidates):
        return self.net(sa)

class Energy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(28, 71), nn.GELU(), nn.Linear(71, 71), nn.GELU(), nn.Linear(71, 1))
    def forward(self, sa, candidates):
        joint = torch.cat((sa[:, None, :].expand(-1, 30, -1), candidates[None, :, :].expand(sa.shape[0], -1, -1)), dim=-1)
        return -self.net(joint).squeeze(-1)

def create(kind, seed):
    torch.manual_seed(seed)
    model = {'direct': Direct, 'energy': Energy}[kind]()
    assert sum(p.numel() for p in model.parameters()) == {'direct': 7198, 'energy': 7243}[kind]
    return model, torch.optim.Adam(model.parameters(), lr=.003, weight_decay=0)
