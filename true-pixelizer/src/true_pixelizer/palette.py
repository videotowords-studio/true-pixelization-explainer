from typing import Sequence, Tuple

import numpy as np
from PIL import Image

from .colors import RGB, srgb_u8_to_oklab
from .errors import ProcessingError


_MAPPING_CHUNK_PIXELS = 4096
_BAYER4 = np.asarray(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
    dtype=np.float64,
)


def _normalize_colors(colors: Sequence[RGB]) -> Tuple[RGB, ...]:
    normalized = []
    seen = set()
    for color in colors:
        if len(color) != 3:
            raise ProcessingError("Palette colors must contain exactly three channels.")
        value = tuple(int(channel) for channel in color)
        if any(channel < 0 or channel > 255 for channel in value):
            raise ProcessingError("Palette color channels must be between 0 and 255.")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return tuple(normalized)  # type: ignore[return-value]


def _pack_rgb(colors: np.ndarray) -> np.ndarray:
    rgb = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
    return (
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32)
    )


def _unique_rgb_with_counts(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    packed, counts = np.unique(_pack_rgb(rgb), return_counts=True)
    colors = np.empty((len(packed), 3), dtype=np.uint8)
    colors[:, 0] = (packed >> 16).astype(np.uint8)
    colors[:, 1] = (packed >> 8).astype(np.uint8)
    colors[:, 2] = packed.astype(np.uint8)
    return colors, counts


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = np.zeros((len(left), len(right)), dtype=np.float64)
    for channel in range(3):
        delta = left[:, channel, None] - right[None, :, channel]
        distances += delta * delta
    return distances


def _median_cut_palette(
    colors: np.ndarray, counts: np.ndarray, color_limit: int
) -> Tuple[RGB, ...]:
    samples = np.repeat(colors, counts.astype(np.intp, copy=False), axis=0)
    sample_image = Image.frombytes(
        "RGB", (len(samples), 1), np.ascontiguousarray(samples).tobytes()
    )
    quantized = sample_image.quantize(
        colors=color_limit,
        method=Image.Quantize.MEDIANCUT,
        kmeans=0,
        dither=Image.Dither.NONE,
    )
    palette_data = quantized.getpalette()
    if palette_data is None:
        raise ProcessingError("Pillow did not return a palette after quantization.")

    indices = np.asarray(quantized, dtype=np.uint8).reshape(-1)
    usage = np.bincount(indices, minlength=256)
    entries = []
    seen = set()
    for palette_index in np.flatnonzero(usage):
        offset = int(palette_index) * 3
        color = tuple(int(value) for value in palette_data[offset : offset + 3])
        if color in seen:
            continue
        seen.add(color)
        entries.append((int(usage[palette_index]), color))

    entries.sort(key=lambda item: (-item[0], item[1][0], item[1][1], item[1][2]))
    return tuple(  # type: ignore[return-value]
        item[1] for item in entries[:color_limit]
    )


def build_palette(
    visible_rgb: np.ndarray,
    max_colors: int,
    fixed_palette: Sequence[RGB],
    locked_colors: Sequence[RGB],
) -> Tuple[RGB, ...]:
    if not 1 <= max_colors <= 256:
        raise ProcessingError("max_colors must be between 1 and 256.")

    locked = _normalize_colors(locked_colors)
    if len(locked) > max_colors:
        raise ProcessingError("Locked colors exceed max_colors.")

    fixed = _normalize_colors(fixed_palette)
    if fixed:
        if len(fixed) > max_colors:
            raise ProcessingError("The fixed palette exceeds max_colors.")
        fixed_keys = set(fixed)
        if any(color not in fixed_keys for color in locked):
            raise ProcessingError("Every locked color must exist in the fixed palette.")
        locked_keys = set(locked)
        return locked + tuple(color for color in fixed if color not in locked_keys)

    colors, counts = _unique_rgb_with_counts(visible_rgb)
    if len(colors) == 0:
        raise ProcessingError("Cannot build a palette from an empty image.")

    locked_keys = set(locked)
    if locked:
        packed_locked = np.fromiter(
            ((red << 16) | (green << 8) | blue for red, green, blue in locked),
            dtype=np.uint32,
            count=len(locked),
        )
        unlocked_mask = ~np.isin(
            _pack_rgb(colors), packed_locked, assume_unique=True
        )
    else:
        unlocked_mask = np.ones(len(colors), dtype=bool)

    unlocked = colors[unlocked_mask]
    unlocked_counts = counts[unlocked_mask]
    remaining_slots = max_colors - len(locked)
    if remaining_slots == 0 or len(unlocked) == 0:
        return locked

    if len(unlocked) <= remaining_slots:
        order = sorted(
            range(len(unlocked)),
            key=lambda index: (
                -int(unlocked_counts[index]),
                int(unlocked[index, 0]),
                int(unlocked[index, 1]),
                int(unlocked[index, 2]),
            ),
        )
        tail = tuple(
            tuple(int(value) for value in unlocked[index]) for index in order
        )
        return locked + tail  # type: ignore[return-value]

    generated = _median_cut_palette(unlocked, unlocked_counts, remaining_slots)
    generated = tuple(color for color in generated if color not in locked_keys)
    return locked + generated


def map_grid_to_palette(
    grid: np.ndarray, palette: Sequence[RGB], dither: str
) -> np.ndarray:
    output = grid.copy()
    palette_values = _normalize_colors(palette)
    if not palette_values:
        raise ProcessingError("Cannot map pixels with an empty palette.")
    if len(palette_values) > 256:
        raise ProcessingError("A palette may contain at most 256 colors.")

    palette_array = np.asarray(palette_values, dtype=np.uint8)
    palette_labs = srgb_u8_to_oklab(palette_array)
    flat_output = output.reshape((-1, 4))
    visible_indices = np.flatnonzero(flat_output[:, 3] > 0)
    width = output.shape[1]

    for start in range(0, len(visible_indices), _MAPPING_CHUNK_PIXELS):
        chunk_indices = visible_indices[start : start + _MAPPING_CHUNK_PIXELS]
        source_labs = srgb_u8_to_oklab(flat_output[chunk_indices, :3])
        distances = _squared_distances(source_labs, palette_labs)
        nearest = np.argmin(distances, axis=1)

        if dither == "bayer4" and len(palette_array) > 1:
            rows = np.arange(len(chunk_indices))
            first = nearest
            first_distance = distances[rows, first].copy()
            distances[rows, first] = np.inf
            second = np.argmin(distances, axis=1)
            second_distance = distances[rows, second]
            probability_second = first_distance / np.maximum(
                first_distance + second_distance, np.finfo(np.float64).eps
            )

            y_coordinates = chunk_indices // width
            x_coordinates = chunk_indices % width
            thresholds = (
                _BAYER4[y_coordinates % 4, x_coordinates % 4] + 0.5
            ) / 16.0
            nearest = first.copy()
            use_second = thresholds < probability_second
            nearest[use_second] = second[use_second]

        flat_output[chunk_indices, :3] = palette_array[nearest]

    flat_output[flat_output[:, 3] == 0, :3] = 0
    return output
