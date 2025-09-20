import torch
import torch.nn as nn


class PaletteDiscriminator(nn.Module):
    """
    Conditional MLP Discriminator.
    Inputs:
      - palette: (B, max_colors, 3) in [-1,1]
      - cond: (B, cond_dim)
    Output:
      - logits: (B,)
    """
    def __init__(self, cond_dim: int = 384, hidden_dim: int = 512, max_colors: int = 7):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.max_colors = max_colors

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            nn.Linear(max_colors * 3 + hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),  # logits
        )

    def forward(self, palette: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        p = palette.view(palette.size(0), -1)
        c = self.cond_proj(cond)
        x = torch.cat([p, c], dim=1)
        logits = self.net(x).squeeze(1)
        return logits
