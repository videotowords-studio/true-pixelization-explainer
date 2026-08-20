from typing import Iterable, Tuple

import numpy as np

from .errors import ConfigurationError

RGB = Tuple[int, int, int]


def parse_hex_color(value: str) -> RGB:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        raise ConfigurationError("Colors must use #RRGGBB or #RGB format.")
    try:
        channels = tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ConfigurationError("Colors must contain hexadecimal digits.") from exc
    return channels  # type: ignore[return-value]


def rgb_to_hex(color: RGB) -> str:
    return "#{:02X}{:02X}{:02X}".format(*color)


def unique_colors(colors: Iterable[RGB]) -> Tuple[RGB, ...]:
    seen = set()
    ordered = []
    for color in colors:
        normalized = tuple(int(channel) for channel in color)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)  # type: ignore[return-value]


def srgb_u8_to_oklab(colors: np.ndarray) -> np.ndarray:
    rgb = np.asarray(colors, dtype=np.float64) / 255.0
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )
    red = linear[..., 0]
    green = linear[..., 1]
    blue = linear[..., 2]

    light = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue

    light_root = np.cbrt(light)
    medium_root = np.cbrt(medium)
    short_root = np.cbrt(short)

    return np.stack(
        (
            0.2104542553 * light_root
            + 0.7936177850 * medium_root
            - 0.0040720468 * short_root,
            1.9779984951 * light_root
            - 2.4285922050 * medium_root
            + 0.4505937099 * short_root,
            0.0259040371 * light_root
            + 0.7827717662 * medium_root
            - 0.8086757660 * short_root,
        ),
        axis=-1,
    )
