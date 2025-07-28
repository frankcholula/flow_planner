import torch
from torch import nn, Tensor
from src.models.activation_fns import Swish


class MLP(nn.Module):
    def __init__(self, input_dim: int, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        original_shape = x.shape
        x = x.view(-1, self.input_dim)
        t = t.unsqueeze(1).float()
        h = torch.cat([x, t], dim=1)
        output = self.main(h)
        return output.view(original_shape)


class CNN(nn.Module):
    def __init__(
        self,
        horizon: int,
        input_dim: int,
        time_dim: int = 1,
        hidden_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        input_channels = input_dim + time_dim
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels, hidden_dim, kernel_size=kernel_size, padding="same"
            ),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=kernel_size, padding="same"),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_reshaped = x.view(-1, self.horizon, self.input_dim).permute(0, 2, 1)
        t_expanded = t.view(-1, 1, 1).expand(-1, 1, self.horizon)
        h = torch.cat([x_reshaped, t_expanded], dim=1)
        out = self.main(h)
        return out.permute(0, 2, 1).reshape(x.shape)


class ConditionalCNN(torch.nn.Module):
    def __init__(
        self,
        horizon: int,
        input_dim: int,
        time_dim: int = 1,
        cond_dim: int = 1,
        hidden_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        input_channels = input_dim + time_dim + cond_dim
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_channels, hidden_dim, kernel_size=kernel_size, padding="same"
            ),
            Swish(),
            torch.nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"
            ),
            Swish(),
            torch.nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"
            ),
            Swish(),
            torch.nn.Conv1d(
                hidden_dim, input_dim, kernel_size=kernel_size, padding="same"
            ),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        x_reshaped = x.view(-1, self.horizon, self.input_dim).permute(0, 2, 1)
        t_expanded = t.view(-1, 1, 1).expand(-1, 1, self.horizon)
        c_expanded = c.view(-1, self.cond_dim, 1).expand(
            -1, self.cond_dim, self.horizon
        )
        h = torch.cat([x_reshaped, t_expanded, c_expanded], dim=1)
        output_reshaped = self.main(h)
        return output_reshaped.permute(0, 2, 1).reshape(x.shape)


# TODO: Implement UNet
class UNet(nn.Module):
    pass


# TODO: Implement ControlNet for conditioning case.
class ControlNet(nn.Module):
    pass
