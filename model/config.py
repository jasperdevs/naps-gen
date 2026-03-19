"""Model configuration dataclass."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the U-Net diffusion model."""

    image_size: int = 64
    image_width: int | None = None
    image_height: int | None = None
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.1
    num_heads: int = 4
    num_timesteps: int = 1000

    @property
    def channel_sizes(self) -> list[int]:
        return [self.base_channels * m for m in self.channel_mult]

    @property
    def sample_width(self) -> int:
        return self.image_width or self.image_size

    @property
    def sample_height(self) -> int:
        return self.image_height or self.image_size

    def to_dict(self) -> dict:
        return {
            "image_size": self.image_size,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "channel_mult": list(self.channel_mult),
            "num_res_blocks": self.num_res_blocks,
            "attention_resolutions": list(self.attention_resolutions),
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "num_timesteps": self.num_timesteps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = d.copy()
        if "channel_mult" in d:
            d["channel_mult"] = tuple(d["channel_mult"])
        if "attention_resolutions" in d:
            d["attention_resolutions"] = tuple(d["attention_resolutions"])
        return cls(**d)


# Preset configs for different scales
SMALL_CONFIG = ModelConfig(
    image_size=64,
    base_channels=64,
    channel_mult=(1, 2, 4, 8),
    num_res_blocks=2,
    attention_resolutions=(8, 16),
    num_heads=4,
)

MEDIUM_CONFIG = ModelConfig(
    image_size=128,
    base_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(16,),
    num_heads=4,
)

LARGE_CONFIG = ModelConfig(
    image_size=256,
    base_channels=128,
    channel_mult=(1, 1, 2, 2, 4, 4),
    num_res_blocks=2,
    attention_resolutions=(16, 32),
    num_heads=8,
)
