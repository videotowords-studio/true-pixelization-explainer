from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .colors import RGB, rgb_to_hex, unique_colors
from .errors import ConfigurationError


@dataclass(frozen=True)
class PixelizeConfig:
    width: int
    height: int
    max_colors: int = 16
    alpha_mode: str = "binary"
    alpha_threshold: float = 0.5
    fit: str = "contain"
    anchor: str = "center"
    trim: str = "alpha"
    padding: int = 0
    dither: str = "none"
    preview_scale: int = 8
    palette: Tuple[RGB, ...] = field(default_factory=tuple)
    locked_colors: Tuple[RGB, ...] = field(default_factory=tuple)
    key_color: Optional[RGB] = None
    key_tolerance: float = 0.0
    max_input_pixels: int = 16_000_000
    max_input_bytes: int = 100_000_000

    def __post_init__(self) -> None:
        if not 1 <= self.width <= 1024 or not 1 <= self.height <= 1024:
            raise ConfigurationError("Target width and height must be between 1 and 1024.")
        if self.width * self.height > 1_048_576:
            raise ConfigurationError("Target image may contain at most 1,048,576 pixels.")
        if not 1 <= self.max_colors <= 256:
            raise ConfigurationError("max_colors must be between 1 and 256.")
        if self.alpha_mode not in ("binary", "preserve"):
            raise ConfigurationError("alpha_mode must be binary or preserve.")
        if not 0.0 <= self.alpha_threshold <= 1.0:
            raise ConfigurationError("alpha_threshold must be between 0 and 1.")
        if self.fit not in ("contain", "cover", "stretch"):
            raise ConfigurationError("fit must be contain, cover, or stretch.")
        if self.anchor not in ("center", "bottom-center"):
            raise ConfigurationError("anchor must be center or bottom-center.")
        if self.trim not in ("alpha", "none"):
            raise ConfigurationError("trim must be alpha or none.")
        if self.padding < 0 or self.padding * 2 >= min(self.width, self.height):
            raise ConfigurationError("padding must leave at least one target pixel per axis.")
        if self.dither not in ("none", "bayer4"):
            raise ConfigurationError("dither must be none or bayer4.")
        if not 1 <= self.preview_scale <= 64:
            raise ConfigurationError("preview_scale must be between 1 and 64.")
        preview_pixels = (
            self.width
            * self.height
            * self.preview_scale
            * self.preview_scale
        )
        if preview_pixels > 16_777_216:
            raise ConfigurationError(
                "The enlarged preview may contain at most 16,777,216 pixels."
            )
        if not 0.0 <= self.key_tolerance <= 441.7:
            raise ConfigurationError("key_tolerance must be between 0 and 441.7.")
        if not 1 <= self.max_input_pixels <= 25_000_000:
            raise ConfigurationError(
                "max_input_pixels must be between 1 and 25,000,000."
            )
        if not 1 <= self.max_input_bytes <= 250_000_000:
            raise ConfigurationError(
                "max_input_bytes must be between 1 and 250,000,000."
            )

        palette = unique_colors(self.palette)
        locked = unique_colors(self.locked_colors)
        for name, colors in (("palette", palette), ("locked_colors", locked)):
            if any(
                len(color) != 3
                or any(channel < 0 or channel > 255 for channel in color)
                for color in colors
            ):
                raise ConfigurationError(
                    "Every {} entry must contain three channels from 0 to 255.".format(
                        name
                    )
                )
        if self.key_color is not None and (
            len(self.key_color) != 3
            or any(channel < 0 or channel > 255 for channel in self.key_color)
        ):
            raise ConfigurationError(
                "key_color must contain three channels from 0 to 255."
            )
        object.__setattr__(self, "palette", palette)
        object.__setattr__(self, "locked_colors", locked)

        if len(palette) > self.max_colors:
            raise ConfigurationError("The fixed palette exceeds max_colors.")
        if len(locked) > self.max_colors:
            raise ConfigurationError("Locked colors exceed max_colors.")
        if palette and any(color not in palette for color in locked):
            raise ConfigurationError("Every locked color must exist in the fixed palette.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "max_colors": self.max_colors,
            "alpha_mode": self.alpha_mode,
            "alpha_threshold": self.alpha_threshold,
            "fit": self.fit,
            "anchor": self.anchor,
            "trim": self.trim,
            "padding": self.padding,
            "dither": self.dither,
            "preview_scale": self.preview_scale,
            "palette": [rgb_to_hex(color) for color in self.palette],
            "locked_colors": [rgb_to_hex(color) for color in self.locked_colors],
            "key_color": rgb_to_hex(self.key_color) if self.key_color else None,
            "key_tolerance": self.key_tolerance,
            "max_input_pixels": self.max_input_pixels,
            "max_input_bytes": self.max_input_bytes,
        }
