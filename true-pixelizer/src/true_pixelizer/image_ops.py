import io
import math
import warnings as python_warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .colors import RGB
from .config import PixelizeConfig
from .errors import InputImageError, ProcessingError


def _open_image(data: bytes, max_pixels: int, mode: str) -> Image.Image:
    if not data:
        raise InputImageError("The input image is empty.")
    try:
        with python_warnings.catch_warnings():
            python_warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                if getattr(opened, "n_frames", 1) != 1:
                    raise InputImageError(
                        "Animated or multi-frame images are not supported by this command."
                    )
                width, height = opened.size
                if width < 1 or height < 1:
                    raise InputImageError("The input image has invalid dimensions.")
                if width * height > max_pixels:
                    raise InputImageError(
                        "The input image exceeds the configured pixel safety limit."
                    )
                oriented = ImageOps.exif_transpose(opened)
                oriented.load()
                return oriented.convert(mode)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InputImageError(
            "The input image exceeds the safe decode limits."
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputImageError("The input file is not a supported image.") from exc


def load_rgba_and_mask(
    image_bytes: bytes,
    config: PixelizeConfig,
    mask_bytes: Optional[bytes] = None,
) -> Tuple[np.ndarray, Dict[str, Any], list]:
    image = _open_image(image_bytes, config.max_input_pixels, "RGBA")
    image_size = image.size
    rgba = np.asarray(image, dtype=np.uint8).copy()
    del image
    original_fully_opaque = bool(np.all(rgba[..., 3] == 255))
    input_had_transparency = bool(np.any(rgba[..., 3] < 255))
    warnings = []

    if config.key_color is not None:
        key = np.asarray(config.key_color, dtype=np.int32)
        flat_rgba = rgba.reshape((-1, 4))
        chunk_pixels = 262_144
        for start in range(0, len(flat_rgba), chunk_pixels):
            chunk = flat_rgba[start : start + chunk_pixels]
            delta = chunk[:, :3].astype(np.int32) - key
            distance_squared = np.sum(delta * delta, axis=1, dtype=np.int64)
            chunk[:, 3][distance_squared <= config.key_tolerance**2] = 0

    if mask_bytes is not None:
        mask = _open_image(mask_bytes, config.max_input_pixels, "L")
        if mask.size != image_size:
            raise InputImageError(
                "The explicit mask must match the oriented input image dimensions."
            )
        mask_array = np.asarray(mask, dtype=np.uint8)
        flat_alpha = rgba[..., 3].reshape(-1)
        flat_mask = mask_array.reshape(-1)
        chunk_pixels = 1_048_576
        for start in range(0, len(flat_alpha), chunk_pixels):
            end = min(len(flat_alpha), start + chunk_pixels)
            alpha_chunk = flat_alpha[start:end].astype(np.uint16)
            mask_chunk = flat_mask[start:end].astype(np.uint16)
            flat_alpha[start:end] = (
                (alpha_chunk * mask_chunk + 127) // 255
            ).astype(np.uint8)
        del mask_array, mask

    if (
        original_fully_opaque
        and mask_bytes is None
        and config.key_color is None
    ):
        warnings.append(
            "The source is fully opaque, so its background is retained. "
            "Use an explicit mask or key color when the background should be removed."
        )

    flat_rgba = rgba.reshape((-1, 4))
    chunk_pixels = 1_048_576
    for start in range(0, len(flat_rgba), chunk_pixels):
        chunk = flat_rgba[start : start + chunk_pixels]
        chunk[chunk[:, 3] == 0, :3] = 0
    visible_mask = rgba[..., 3] > 0
    if not np.any(visible_mask):
        raise InputImageError("No visible pixels remain after alpha and mask processing.")

    if config.trim == "alpha":
        bbox_mask = visible_mask
        if config.alpha_mode == "binary":
            threshold_u8 = int(math.ceil(config.alpha_threshold * 255.0))
            bbox_mask = visible_mask & (rgba[..., 3] >= threshold_u8)
            if not np.any(bbox_mask):
                raise InputImageError(
                    "No source pixels meet the requested binary alpha threshold."
                )
        visible_rows = np.flatnonzero(np.any(bbox_mask, axis=1))
        visible_columns = np.flatnonzero(np.any(bbox_mask, axis=0))
        y0, y1 = int(visible_rows[0]), int(visible_rows[-1]) + 1
        x0, x1 = int(visible_columns[0]), int(visible_columns[-1]) + 1
    else:
        y0, x0 = 0, 0
        y1, x1 = rgba.shape[0], rgba.shape[1]

    metadata = {
        "input_size": [int(rgba.shape[1]), int(rgba.shape[0])],
        "source_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "mask_used": mask_bytes is not None,
        "key_color_used": config.key_color is not None,
        "input_had_transparency": input_had_transparency,
    }
    subject = rgba[y0:y1, x0:x1]
    if subject.shape[:2] != rgba.shape[:2]:
        subject = subject.copy()
    return subject, metadata, warnings


def _area_resize_horizontal(source: np.ndarray, out_width: int) -> np.ndarray:
    source = np.asarray(source)
    in_height, in_width = source.shape
    x_edges = np.linspace(0.0, float(in_width), out_width + 1, dtype=np.float64)
    x_index = np.floor(x_edges).astype(np.int64)
    x_fraction = x_edges - x_index
    x_widths = np.diff(x_edges)

    horizontal = np.empty((in_height, out_width), dtype=np.float64)
    chunk_rows = max(1, min(256, in_height))
    for row_start in range(0, in_height, chunk_rows):
        row_end = min(in_height, row_start + chunk_rows)
        block = source[row_start:row_end].astype(np.float64, copy=False)
        cumulative = np.empty((row_end - row_start, in_width + 1), dtype=np.float64)
        cumulative[:, 0] = 0.0
        np.cumsum(block, axis=1, dtype=np.float64, out=cumulative[:, 1:])
        boundaries = cumulative[:, np.minimum(x_index, in_width)].copy()
        valid = x_index < in_width
        if np.any(valid):
            boundaries[:, valid] += (
                block[:, x_index[valid]] * x_fraction[valid][None, :]
            )
        horizontal[row_start:row_end] = (
            boundaries[:, 1:] - boundaries[:, :-1]
        ) / x_widths[None, :]

    return horizontal


def _nearest_resize_horizontal(source: np.ndarray, out_width: int) -> np.ndarray:
    in_width = source.shape[1]
    x_indices = (
        (2 * np.arange(out_width, dtype=np.int64) + 1) * in_width
    ) // (2 * out_width)
    return source[:, x_indices]


def _resize_scalar_axis_aware(
    channel: np.ndarray, out_width: int, out_height: int
) -> np.ndarray:
    source = np.asarray(channel)
    in_height, in_width = source.shape
    horizontal_first_cost = in_height * out_width
    vertical_first_cost = out_height * in_width
    if horizontal_first_cost <= vertical_first_cost:
        resized = _resize_horizontal_axis(source, out_width)
        resized = _resize_horizontal_axis(resized.T, out_height).T
    else:
        resized = _resize_horizontal_axis(source.T, out_height).T
        resized = _resize_horizontal_axis(resized, out_width)
    return resized.astype(np.float64, copy=False)


def _resize_horizontal_axis(source: np.ndarray, out_width: int) -> np.ndarray:
    if out_width < source.shape[1]:
        return _area_resize_horizontal(source, out_width)
    if out_width > source.shape[1]:
        return _nearest_resize_horizontal(source, out_width)
    return source


def _axis_aware_resize_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    source_alpha = rgba[..., 3]
    resized_alpha = _resize_scalar_axis_aware(source_alpha, width, height)
    resized_rgb = np.zeros((height, width, 3), dtype=np.float64)
    for channel_index in range(3):
        premultiplied = np.multiply(
            rgba[..., channel_index], source_alpha, dtype=np.uint16
        )
        resized_premultiplied = _resize_scalar_axis_aware(
            premultiplied, width, height
        )
        np.divide(
            resized_premultiplied,
            resized_alpha,
            out=resized_rgb[..., channel_index],
            where=resized_alpha > 0.0,
        )
    output = np.zeros((height, width, 4), dtype=np.float64)
    output[..., :3] = np.clip(resized_rgb, 0.0, 255.0)
    output[..., 3] = np.clip(resized_alpha, 0.0, 255.0)
    return output


def _nearest_resize_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = rgba.shape[:2]
    x_indices = (
        (2 * np.arange(width, dtype=np.int64) + 1) * source_width
    ) // (2 * width)
    y_indices = (
        (2 * np.arange(height, dtype=np.int64) + 1) * source_height
    ) // (2 * height)
    return rgba[y_indices[:, None], x_indices[None, :]].astype(
        np.float64, copy=False
    )


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def _crop_for_cover(
    rgba: np.ndarray, target_width: int, target_height: int, anchor: str
) -> np.ndarray:
    height, width = rgba.shape[:2]
    if width * target_height > height * target_width:
        crop_width = max(
            1,
            min(
                width,
                _round_half_up_ratio(height * target_width, target_height),
            ),
        )
        x0 = (width - crop_width) // 2
        return rgba[:, x0 : x0 + crop_width]
    crop_height = max(
        1,
        min(
            height,
            _round_half_up_ratio(width * target_height, target_width),
        ),
    )
    if anchor == "bottom-center":
        y0 = height - crop_height
    else:
        y0 = (height - crop_height) // 2
    return rgba[y0 : y0 + crop_height]


def reconstruct_grid(
    subject_rgba: np.ndarray, config: PixelizeConfig
) -> Tuple[np.ndarray, Dict[str, Any], list]:
    available_width = config.width - 2 * config.padding
    available_height = config.height - 2 * config.padding
    source = subject_rgba
    warnings = []

    if config.fit == "cover":
        source = _crop_for_cover(
            source, available_width, available_height, config.anchor
        )
        target_width = available_width
        target_height = available_height
    elif config.fit == "stretch":
        target_width = available_width
        target_height = available_height
    else:
        source_height, source_width = source.shape[:2]
        if available_width * source_height <= available_height * source_width:
            target_width = available_width
            target_height = max(
                1,
                min(
                    available_height,
                    _round_half_up_ratio(
                        source_height * available_width, source_width
                    ),
                ),
            )
        else:
            target_height = available_height
            target_width = max(
                1,
                min(
                    available_width,
                    _round_half_up_ratio(
                        source_width * available_height, source_height
                    ),
                ),
            )

    if target_width > source.shape[1] or target_height > source.shape[0]:
        warnings.append(
            "The source subject is smaller than its target placement on at least one axis."
        )

    x_sampling = (
        "area"
        if target_width < source.shape[1]
        else "nearest"
        if target_width > source.shape[1]
        else "identity"
    )
    y_sampling = (
        "area"
        if target_height < source.shape[0]
        else "nearest"
        if target_height > source.shape[0]
        else "identity"
    )
    if x_sampling != "area" and y_sampling != "area":
        sampled = _nearest_resize_rgba(source, target_width, target_height)
        sampling_method = "nearest-neighbor-center-v1"
    else:
        sampled = _axis_aware_resize_rgba(source, target_width, target_height)
        sampling_method = (
            "premultiplied-alpha-axis-aware-v1:x-{},y-{}".format(
                x_sampling, y_sampling
            )
        )
    grid = np.zeros((config.height, config.width, 4), dtype=np.uint8)
    x_offset = (config.width - target_width) // 2
    if config.anchor == "bottom-center":
        y_offset = config.height - config.padding - target_height
    else:
        y_offset = (config.height - target_height) // 2

    alpha_float = sampled[..., 3] / 255.0
    if config.alpha_mode == "binary":
        alpha = np.where(
            (alpha_float > 0.0) & (alpha_float >= config.alpha_threshold),
            255,
            0,
        ).astype(np.uint8)
    else:
        alpha = np.floor(sampled[..., 3] + 0.5).astype(np.uint8)

    rgb = np.floor(sampled[..., :3] + 0.5).astype(np.uint8)
    rgb[alpha == 0] = 0
    grid[y_offset : y_offset + target_height, x_offset : x_offset + target_width, :3] = rgb
    grid[y_offset : y_offset + target_height, x_offset : x_offset + target_width, 3] = alpha

    if not np.any(grid[..., 3] > 0):
        raise ProcessingError(
            "The alpha threshold removed every target pixel; lower the threshold "
            "or increase the target size."
        )

    metadata = {
        "placement": [x_offset, y_offset, target_width, target_height],
        "sampled_source_size": [int(source.shape[1]), int(source.shape[0])],
        "sampling_method": sampling_method,
    }
    return grid, metadata, warnings
