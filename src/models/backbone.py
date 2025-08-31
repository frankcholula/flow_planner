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
        fusion_strategy: str = "concat",
    ):
        super().__init__()
        self.horizon = horizon
        self.fusion_strategy = fusion_strategy
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

        if self.fusion_strategy == "concat":
            self.feature_projection = nn.Linear(hidden_dim * 2, hidden_dim)

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

        final_emb = t_emb
        if c is not None:
            c_emb = self.cond_embedding(c)
            if self.fusion_strategy == "add":
                final_emb = t_emb + c_emb
            elif self.fusion_strategy == "concat":
                combined = torch.cat([t_emb, c_emb], dim=-1)
                final_emb = self.feature_projection(combined)
            else:
                raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        h = h + rearrange(final_emb, "b d -> b d 1")
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
        cond_dim: int = 8,
        hidden_dim: int = 128,
        fusion_strategy: str = "concat",
        use_mlp_embedding: bool = False,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.horizon = horizon
        self.cond_dim = cond_dim
        self.fusion_strategy = fusion_strategy

        assert input_dim % horizon == 0, "input_dim must be divisible by horizon"
        self.transition_dim = input_dim // horizon

        if use_mlp_embedding:
            print("Using MLP embeddings...")
            time_embed_dim = hidden_dim * expansion_factor
            self.time_embedding = nn.Sequential(
                SinusoidalPosEmb(hidden_dim),
                nn.Linear(hidden_dim, time_embed_dim),
                Mish(),
                nn.Linear(time_embed_dim, hidden_dim),
            )
            cond_embed_dim = hidden_dim * expansion_factor
            self.cond_embedding = nn.Sequential(
                nn.Linear(cond_dim, cond_embed_dim),
                Mish(),
                nn.Linear(cond_embed_dim, hidden_dim),
            )
        else:
            print("Using simple embeddings...")
            self.time_embedding = SinusoidalPosEmb(hidden_dim)
            self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        if self.fusion_strategy == "concat":
            self.feature_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.initial_conv = nn.Conv1d(self.transition_dim, hidden_dim, kernel_size=1)
        self.down1 = DownBlock(hidden_dim, hidden_dim * 2)
        self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4)
        self.bottleneck = ResidualBlock(hidden_dim * 4, hidden_dim * 4)
        self.up1 = UpBlock(
            in_channels=hidden_dim * 4,
            skip_channels=hidden_dim * 4,
            out_channels=hidden_dim * 2,
        )
        self.up2 = UpBlock(
            in_channels=hidden_dim * 2,
            skip_channels=hidden_dim * 2,
            out_channels=hidden_dim,
        )
        self.final_conv = nn.Conv1d(hidden_dim, self.transition_dim, kernel_size=1)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        x_initial = self.initial_conv(rearrange(x, "b (h d) -> b d h", h=self.horizon))

        t_float = t.float()
        t_scaled = t_float * 1000.0 if t_float.max() <= 1.0 else t_float
        t_emb = self.time_embedding(t_scaled)

        if c is not None:
            c_emb = self.cond_embedding(c.float())

            if self.fusion_strategy == "concat":
                combined_emb = torch.cat([t_emb, c_emb], dim=-1)
                final_emb = self.feature_projection(combined_emb)
            elif self.fusion_strategy == "add":
                final_emb = t_emb + c_emb
            else:
                raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        else:
            final_emb = t_emb

        time_cond_emb = repeat(final_emb, "b d -> b d h", h=self.horizon)
        h = x_initial + time_cond_emb

        skip1, h = self.down1(h)
        skip2, h = self.down2(h)
        h = self.bottleneck(h)
        h = self.up1(h, skip2)
        h = self.up2(h, skip1)

        output_reshaped = self.final_conv(h)
        output_flat = rearrange(output_reshaped, "b d h -> b (h d)")
        return output_flat


# class ConditionalUNet1D(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         horizon: int,
#         cond_dim: int,
#         hidden_dim: int = 128,
#         time_dim: Optional[int] = None,
#         fusion_strategy: str = "concat",
#         use_mlp_embedding: bool = False,
#         expansion_factor: int = 4
#     ):
#         super().__init__()
#         self.horizon = horizon
#         self.fusion_strategy = fusion_strategy
#         assert input_dim % horizon == 0, "input_dim must be divisible by horizon"
#         self.transition_dim = input_dim // horizon

#         if time_dim is None:
#             time_dim = hidden_dim

#         if use_mlp_embedding:
#             # warning: this is slow
#             print("Using MLP embedding...")
#             embedding_dim = hidden_dim * expansion_factor
#             self.time_embedding = nn.Sequential(
#                 SinusoidalPosEmb(time_dim),
#                 nn.Linear(time_dim, embedding_dim),
#                 Mish(),
#                 nn.Linear(embedding_dim, hidden_dim),
#             )
#             self.cond_embedding = nn.Sequential(
#                 nn.Linear(cond_dim, embedding_dim),
#                 Mish(),
#                 nn.Linear(embedding_dim, hidden_dim),
#             )
#         else:
#             print("Using simple linear embedding...")
#             self.time_embedding = SinusoidalPosEmb(hidden_dim)
#             self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

#         if self.fusion_strategy == "concat":
#             self.feature_projection = nn.Linear(hidden_dim * 2, hidden_dim)

#         # U-Net architecture remains the same
#         self.initial_conv = nn.Conv1d(self.transition_dim, hidden_dim, kernel_size=1)
#         self.down1 = DownBlock(hidden_dim, hidden_dim * 2)
#         self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4)
#         self.bottleneck = ResidualBlock(hidden_dim * 4, hidden_dim * 4)
#         self.up1 = UpBlock(hidden_dim * 4, hidden_dim * 4, hidden_dim * 2)
#         self.up2 = UpBlock(hidden_dim * 2, hidden_dim * 2, hidden_dim)
#         self.final_conv = nn.Conv1d(hidden_dim, self.transition_dim, kernel_size=1)

#     def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
#         x_reshaped = rearrange(x, "b (h d) -> b d h", h=self.horizon)
#         h = self.initial_conv(x_reshaped)

#         t_float = t.float()
#         t_scaled = t_float * 1000.0 if t_float.max() <= 1.0 else t_float
#         t_emb = self.time_embedding(t_scaled)

#         final_emb = t_emb
#         if c is not None:
#             c_emb = self.cond_embedding(c)
#             if self.fusion_strategy == "add":
#                 final_emb = t_emb + c_emb
#             elif self.fusion_strategy == "concat":
#                 combined = torch.cat([t_emb, c_emb], dim=-1)
#                 final_emb = self.feature_projection(combined)
#             else:
#                 raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

#         h = h + rearrange(final_emb, "b d -> b d 1")

#         skip1, h = self.down1(h)
#         skip2, h = self.down2(h)
#         h = self.bottleneck(h)
#         h = self.up1(h, skip2)
#         h = self.up2(h, skip1)

#         out_reshaped = self.final_conv(h)

#         return rearrange(out_reshaped, "b d h -> b (h d)")
