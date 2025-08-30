import torch
from torch import nn, Tensor
from einops import rearrange, repeat
from src.models.positional_embedding import SinusoidalPosEmb
from typing import Optional


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


class Mish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 128, time_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim

        if time_dim is None:
            time_dim = hidden_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.initial_projection = nn.Linear(input_dim, hidden_dim)

        self.main = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        original_shape = x.shape
        x = x.view(-1, self.input_dim)
        x_proj = self.initial_projection(x)
        t_float = t.float()
        t_scaled = t_float * 1000.0 if t_float.max() <= 1.0 else t_float
        t_emb = self.time_embedding(t_scaled)
        h = x_proj + t_emb
        output = self.main(h)
        return output.view(original_shape)


class CNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        kernel_size: int = 5,
        hidden_dim: int = 128,
        time_dim: Optional[int] = None,
    ):
        super().__init__()
        self.horizon = horizon
        assert input_dim % horizon == 0, "input_dim must be divisible by horizon"
        self.transition_dim = input_dim // horizon

        if time_dim is None:
            time_dim = hidden_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.initial_conv = nn.Conv1d(self.transition_dim, hidden_dim, kernel_size=1)
        self.main = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
        )
        self.final_conv = nn.Conv1d(hidden_dim, self.transition_dim, kernel_size=1)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        x_reshaped = rearrange(x, "b (h d) -> b d h", h=self.horizon)
        h = self.initial_conv(x_reshaped)

        t_float = t.float()
        t_scaled = t_float * 1000.0 if t_float.max() <= 1.0 else t_float
        t_emb = self.time_embedding(t_scaled)

        h = h + rearrange(t_emb, "b d -> b d 1")
        h = self.main(h)
        out_reshaped = self.final_conv(h)
        return rearrange(out_reshaped, "b d h -> b (h d)")


class ConditionalCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        cond_dim: int,
        hidden_dim: int = 128,
        time_dim: Optional[int] = None,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        assert input_dim % horizon == 0, "input_dim must be divisible by horizon"
        self.transition_dim = input_dim // horizon

        if time_dim is None:
            time_dim = hidden_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.initial_conv = nn.Conv1d(self.transition_dim, hidden_dim, kernel_size=1)
        self.main = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
        )
        self.final_conv = nn.Conv1d(hidden_dim, self.transition_dim, kernel_size=1)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        x_reshaped = rearrange(x, "b (h d) -> b d h", h=self.horizon)
        h = self.initial_conv(x_reshaped)

        t_float = t.float()
        t_scaled = t_float * 1000.0 if t_float.max() <= 1.0 else t_float
        t_emb = self.time_embedding(t_scaled)
        h = h + rearrange(t_emb, "b d -> b d 1")

        if c is not None:
            c_emb = self.cond_embedding(c)
            h = h + rearrange(c_emb, "b d -> b d 1")
        h = self.main(h)
        out_reshaped = self.final_conv(h)
        return rearrange(out_reshaped, "b d h -> b (h d)")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.activation = Swish()
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.downsample = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        x = self.res_block(x)
        return x, self.downsample(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # The upsample layer should reduce the channels to match the desired output level
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # The res_block takes the upsampled channels + the skip connection channels
        self.res_block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        # Pad if necessary to handle potential off-by-one errors from strided conv
        if x.shape[-1] != skip_connection.shape[-1]:
            padding = skip_connection.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (padding, 0))
        x = torch.cat([x, skip_connection], dim=1)
        return self.res_block(x)


class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        cond_dim: int,
        hidden_dim: int = 128,
        time_dim: Optional[int] = None,
    ):
        super().__init__()
        self.horizon = horizon
        assert input_dim % horizon == 0, "input_dim must be divisible by horizon"
        self.transition_dim = input_dim // horizon

        if time_dim is None:
            time_dim = hidden_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.initial_conv = nn.Conv1d(self.transition_dim, hidden_dim, kernel_size=1)
        self.down1 = DownBlock(hidden_dim, hidden_dim * 2)
        self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4)
        self.bottleneck = ResidualBlock(hidden_dim * 4, hidden_dim * 4)
        self.up1 = UpBlock(hidden_dim * 4, hidden_dim * 4, hidden_dim * 2)
        self.up2 = UpBlock(hidden_dim * 2, hidden_dim * 2, hidden_dim)
        self.final_conv = nn.Conv1d(hidden_dim, self.transition_dim, kernel_size=1)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        x_reshaped = rearrange(x, "b (h d) -> b d h", h=self.horizon)
        h = self.initial_conv(x_reshaped)

        t_float = t.float()
        t_scaled = t_float * 1000.0 if t_float.max() <= 1.0 else t_float
        t_emb = self.time_embedding(t_scaled)
        h = h + rearrange(t_emb, "b d -> b d 1")

        if c is not None:
            c_emb = self.cond_embedding(c)
            h = h + rearrange(c_emb, "b d -> b d 1")

        skip1, h = self.down1(h)
        skip2, h = self.down2(h)
        h = self.bottleneck(h)
        h = self.up1(h, skip2)
        h = self.up2(h, skip1)

        out_reshaped = self.final_conv(h)
        return rearrange(out_reshaped, "b d h -> b (h d)")
