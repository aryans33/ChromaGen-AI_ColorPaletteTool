import torch
import torch.nn as nn


class PaletteGenerator(nn.Module):
    """
    Conditional MLP Generator.
    Inputs:
      - z: noise (B, z_dim)
      - cond: text embedding (B, cond_dim)
    Output:
      - palette in [-1, 1], shape (B, max_colors, 3)
    """
    def __init__(self, z_dim: int = 100, cond_dim: int = 384, hidden_dim: int = 512, max_colors: int = 7):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.max_colors = max_colors

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, max_colors * 3),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.cond_proj(cond)
        x = torch.cat([z, c], dim=1)
        out = self.net(x)
        return out.view(-1, self.max_colors, 3)
