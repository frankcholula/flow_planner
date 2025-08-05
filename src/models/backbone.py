import torch
from torch import nn, Tensor
from einops import rearrange, repeat
from src.models.positional_embedding import SinusoidalPosEmb


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


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
        input_dim: int,
        horizon: int,
        time_dim: int = 1,
        hidden_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.horizon = horizon

        # calculate the transition dim
        assert input_dim % horizon == 0, "input_dim must be divisible by horizon"
        self.transition_dim = input_dim // horizon

        input_channels = self.transition_dim + time_dim
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels, hidden_dim, kernel_size=kernel_size, padding="same"
            ),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            Swish(),
            nn.Conv1d(
                hidden_dim, self.transition_dim, kernel_size=kernel_size, padding="same"
            ),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_reshaped = x.view(-1, self.horizon, self.transition_dim).permute(0, 2, 1)
        t_expanded = t.view(-1, 1, 1).expand(-1, 1, self.horizon)
        h = torch.cat([x_reshaped, t_expanded], dim=1)
        out = self.main(h)
        return out.permute(0, 2, 1).reshape(x.shape)


class ConditionalCNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        time_dim: int = 1,
        cond_dim: int = 1,
        hidden_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        self.cond_dim = cond_dim

        assert input_dim % horizon == 0, "input_dim must be divisible by horizon"
        self.transition_dim = input_dim // horizon

        input_channels = self.transition_dim + time_dim + cond_dim
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
                hidden_dim, self.transition_dim, kernel_size=kernel_size, padding="same"
            ),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        x_reshaped = x.view(-1, self.horizon, self.transition_dim).permute(0, 2, 1)
        t_expanded = t.view(-1, 1, 1).expand(-1, 1, self.horizon)
        c_expanded = c.view(-1, self.cond_dim, 1).expand(
            -1, self.cond_dim, self.horizon
        )
        h = torch.cat([x_reshaped, t_expanded, c_expanded], dim=1)
        output_reshaped = self.main(h)
        return output_reshaped.permute(0, 2, 1).reshape(x.shape)


# diffusers doesn't have a conditional Unet1D, so we implement our own.
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.res_block = ResidualBlock(out_channels * 2, out_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.res_block(x)


class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        cond_dim: int = 1,
        hidden_dim: int = 128,
        fusion_strategy: str = "concat",
    ):
        super().__init__()
        self.horizon = horizon
        self.cond_dim = cond_dim

        # Internally calculate the transition_dim (channels)
        assert input_dim % horizon == 0, "input_dim must be divisible by horizon"
        self.transition_dim = input_dim // horizon

        # Embeddings for flow time
        time_embed_dim = hidden_dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, hidden_dim),
        )

        # Embedding for condition, just something easy for now
        # TODO: can do something more sophisticated later.
        self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        if fusion_strategy == "concat":
            self.feature_projection = nn.Linear(hidden_dim * 2, hidden_dim)
            pass

        # Initial convolution to map input to hidden dimension
        self.initial_conv = nn.Conv1d(self.transition_dim, hidden_dim, kernel_size=1)

        # Downsampling Path
        self.down1 = DownBlock(hidden_dim, hidden_dim * 2)
        self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4)

        # Bottleneck
        self.bottleneck = ResidualBlock(hidden_dim * 4, hidden_dim * 4)

        # Upsampling Path
        self.up1 = UpBlock(hidden_dim * 4, hidden_dim * 2)
        self.up2 = UpBlock(hidden_dim * 2, hidden_dim)

        # Final convolution to map back to the original transition dimension
        self.final_conv = nn.Conv1d(hidden_dim, self.transition_dim, kernel_size=1)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        x_reshaped = rearrange(x, "b (h d) -> b d h", h=self.horizon)
        x_initial = self.initial_conv(x_reshaped)

        # embed time and condition
        t_emb = self.time_embedding(t.float().unsqueeze(1))
        c_emb = self.cond_embedding(c.float())

        if self.fusion_strategy == "concat":
            combined_emb = torch.cat([t_emb, c_emb], dim=-1)
            final_emb = self.feature_projection(combined_emb)
        elif self.fusion_strategy == "add":
            final_emb = t_emb + c_emb
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        

        time_cond_emb = repeat(t_emb + c_emb, "b d -> b d h", h=self.horizon)
        h = x_initial + time_cond_emb

        # 3. U-Net Path
        skip1, h = self.down1(h)
        skip2, h = self.down2(h)

        h = self.bottleneck(h)

        h = self.up1(h, skip2)
        h = self.up2(h, skip1)

        # 4. Final Layer and Reshape
        output_reshaped = self.final_conv(h)
        output_flat = rearrange(output_reshaped, "b d h -> b (h d)")

        return output_flat


# TODO: Implement ControlNet for conditioning case.
class ControlNet(nn.Module):
    pass
