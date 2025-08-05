import torch
from torch import nn, Tensor
import math
from einops import rearrange, repeat


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class SinusoidalPosEmb(nn.Module):
    """A sinusoidal time embedding layer."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers."""

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


class Downsample1d(nn.Module):
    """A learnable downsampling layer with a strided convolution."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """A learnable upsampling layer with a transposed convolution."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# --- The Main U-Net Model ---
class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        cond_dim: int = 1,
        hidden_dims=[128, 256, 512],  # Example hidden dimensions
    ):
        super().__init__()
        self.horizon = horizon

        assert input_dim % horizon == 0
        self.transition_dim = input_dim // horizon

        # Time and Condition Embeddings
        time_embed_dim = hidden_dims[0] * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(hidden_dims[0]),
            nn.Linear(hidden_dims[0], time_embed_dim),
            Mish(),
            nn.Linear(time_embed_dim, hidden_dims[0]),
        )
        self.cond_embedding = nn.Linear(cond_dim, hidden_dims[0])

        self.initial_conv = nn.Conv1d(
            self.transition_dim, hidden_dims[0], kernel_size=1
        )

        # --- Downsampling Path ---
        self.down_modules = nn.ModuleList()
        all_dims = [hidden_dims[0]] + hidden_dims
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        for dim_in, dim_out in in_out:
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ResidualBlock(dim_in, dim_out),
                        ResidualBlock(dim_out, dim_out),
                        Downsample1d(dim_out),
                    ]
                )
            )

        # --- Bottleneck ---
        mid_dim = hidden_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ResidualBlock(mid_dim, mid_dim),
                ResidualBlock(mid_dim, mid_dim),
            ]
        )

        # --- Upsampling Path ---
        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ResidualBlock(dim_out * 2, dim_in),  # x2 for skip connection
                        ResidualBlock(dim_in, dim_in),
                        Upsample1d(dim_in),
                    ]
                )
            )

        self.final_conv = nn.Conv1d(hidden_dims[0], self.transition_dim, 1)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        # 1. Initial processing and embedding
        x = self.initial_conv(rearrange(x, "b (h d) -> b d h", h=self.horizon))
        t_emb = self.time_embedding(t.float())
        c_emb = self.cond_embedding(c.float())

        # Additive fusion
        emb = repeat(t_emb + c_emb, "b d -> b d h", h=x.shape[-1])
        x = x + emb

        # 2. Downsampling path
        skip_connections = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x)
            x = resnet2(x)
            skip_connections.append(x)
            x = downsample(x)

        # 3. Bottleneck
        for mid_module in self.mid_modules:
            x = mid_module(x)

        # 4. Upsampling path
        for resnet1, resnet2, upsample in self.up_modules:
            skip = skip_connections.pop()
            # Pad if there's a size mismatch from downsampling
            if x.shape[-1] != skip.shape[-1]:
                padding = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, (padding, 0))

            x = torch.cat((x, skip), dim=1)
            x = resnet1(x)
            x = resnet2(x)
            x = upsample(x)

        # 5. Final output
        x = self.final_conv(x)
        return rearrange(x, "b d h -> b (h d)")
