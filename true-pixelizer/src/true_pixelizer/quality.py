import io
import warnings
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .colors import RGB
from .config import PixelizeConfig
from .errors import InputImageError


def _effective_palette(
    config: PixelizeConfig, allowed_palette: Sequence[RGB]
) -> Tuple[Tuple[RGB, ...], bool]:
    try:
        source = allowed_palette if len(allowed_palette) > 0 else config.palette
        normalized = []
        for color in source:
            if len(color) != 3:
                return tuple(), False
            channels = tuple(int(channel) for channel in color)
            if any(channel < 0 or channel > 255 for channel in channels):
                return tuple(), False
            normalized.append(channels)
        return tuple(normalized), True  # type: ignore[return-value]
    except (TypeError, ValueError, OverflowError):
        return tuple(), False


def _base_metrics(array: np.ndarray, config: PixelizeConfig) -> Dict[str, Any]:
    shape = list(array.shape)
    actual_height = int(array.shape[0]) if array.ndim >= 1 else None
    actual_width = int(array.shape[1]) if array.ndim >= 2 else None
    return {
        "input_shape": shape,
        "input_dtype": str(array.dtype),
        "expected_dtype": "uint8",
        "actual_width": actual_width,
        "actual_height": actual_height,
        "expected_width": config.width,
        "expected_height": config.height,
        "visible_pixel_count": 0,
        "visible_color_count": 0,
        "max_colors": config.max_colors,
        "invalid_alpha_pixel_count": 0,
        "transparent_rgb_pixel_count": 0,
        "allowed_palette_color_count": 0,
        "invalid_palette_color_count": 0,
        "invalid_palette_pixel_count": 0,
    }


def _all_checks_pass(checks: Dict[str, Dict[str, Any]]) -> bool:
    return all(bool(check.get("passed", False)) for check in checks.values())


def validate_rgba(
    rgba: np.ndarray,
    config: PixelizeConfig,
    allowed_palette: Sequence[RGB] = (),
) -> Dict[str, Any]:
    """Return hard quality results without modifying the supplied pixel array."""
    try:
        array = np.asarray(rgba)
    except Exception:
        array = np.empty((0,), dtype=np.uint8)

    metrics = _base_metrics(array, config)
    palette, palette_is_valid = _effective_palette(config, allowed_palette)
    metrics["allowed_palette_color_count"] = len(palette)
    actual_size = [metrics["actual_width"], metrics["actual_height"]]
    checks: Dict[str, Dict[str, Any]] = {
        "dimensions": {
            "passed": False,
            "expected_size": [config.width, config.height],
            "actual_size": actual_size,
        },
        "pixel_format": {
            "passed": False,
            "expected": "uint8 RGBA",
            "actual_dtype": str(array.dtype),
            "actual_shape": list(array.shape),
        },
        "visible_pixels": {"passed": False, "actual": 0, "minimum": 1},
        "color_count": {
            "passed": False,
            "actual": 0,
            "maximum": config.max_colors,
        },
        "binary_alpha": {
            "passed": False,
            "applicable": config.alpha_mode == "binary",
            "invalid_pixel_count": 0,
        },
        "transparent_rgb": {"passed": False, "invalid_pixel_count": 0},
        "allowed_palette": {
            "passed": False,
            "applicable": bool(palette),
            "allowed_color_count": len(palette),
            "invalid_color_count": 0,
            "invalid_pixel_count": 0,
        },
    }

    is_rgba = array.ndim == 3 and array.shape[2] == 4
    is_uint8 = array.dtype == np.dtype(np.uint8)
    checks["dimensions"]["passed"] = bool(
        is_rgba
        and array.shape[0] == config.height
        and array.shape[1] == config.width
    )
    checks["pixel_format"]["passed"] = bool(is_rgba and is_uint8)
    if not is_rgba or not is_uint8:
        return {"passed": False, "checks": checks, "metrics": metrics}

    try:
        alpha = array[..., 3]
        rgb = array[..., :3]
        visible_mask = alpha > 0
        transparent_mask = alpha == 0
        visible_pixel_count = int(np.count_nonzero(visible_mask))
        metrics["visible_pixel_count"] = visible_pixel_count
        checks["visible_pixels"].update(
            {"passed": visible_pixel_count > 0, "actual": visible_pixel_count}
        )

        if visible_pixel_count > 0:
            visible_colors, visible_counts = np.unique(
                rgb[visible_mask], axis=0, return_counts=True
            )
        else:
            visible_colors = np.empty((0, 3), dtype=array.dtype)
            visible_counts = np.empty((0,), dtype=np.int64)

        visible_color_count = int(len(visible_colors))
        metrics["visible_color_count"] = visible_color_count
        checks["color_count"].update(
            {
                "passed": visible_color_count <= config.max_colors,
                "actual": visible_color_count,
            }
        )

        if config.alpha_mode == "binary":
            invalid_alpha = (alpha != 0) & (alpha != 255)
            invalid_alpha_count = int(np.count_nonzero(invalid_alpha))
            metrics["invalid_alpha_pixel_count"] = invalid_alpha_count
            checks["binary_alpha"].update(
                {
                    "passed": invalid_alpha_count == 0,
                    "invalid_pixel_count": invalid_alpha_count,
                }
            )
        else:
            checks["binary_alpha"]["passed"] = True

        hidden_rgb = np.any(rgb[transparent_mask] != 0, axis=1)
        hidden_rgb_count = int(np.count_nonzero(hidden_rgb))
        metrics["transparent_rgb_pixel_count"] = hidden_rgb_count
        checks["transparent_rgb"].update(
            {
                "passed": hidden_rgb_count == 0,
                "invalid_pixel_count": hidden_rgb_count,
            }
        )

        if not palette_is_valid:
            checks["allowed_palette"]["palette_valid"] = False
        elif not palette:
            checks["allowed_palette"].update(
                {"passed": True, "applicable": False, "palette_valid": True}
            )
        else:
            allowed = set(palette)
            invalid_color_count = 0
            invalid_pixel_count = 0
            for color, count in zip(visible_colors, visible_counts):
                key = tuple(channel.item() for channel in color)
                if key not in allowed:
                    invalid_color_count += 1
                    invalid_pixel_count += int(count)
            metrics["invalid_palette_color_count"] = invalid_color_count
            metrics["invalid_palette_pixel_count"] = invalid_pixel_count
            checks["allowed_palette"].update(
                {
                    "passed": invalid_color_count == 0,
                    "palette_valid": True,
                    "invalid_color_count": invalid_color_count,
                    "invalid_pixel_count": invalid_pixel_count,
                }
            )
    except (TypeError, ValueError, OverflowError):
        return {"passed": False, "checks": checks, "metrics": metrics}

    return {
        "passed": _all_checks_pass(checks),
        "checks": checks,
        "metrics": metrics,
    }


def _check_png_header(image: Image.Image, max_pixels: int) -> None:
    if image.format != "PNG":
        raise InputImageError("The input file is not a PNG image.")
    width, height = image.size
    if width < 1 or height < 1:
        raise InputImageError("The PNG image has invalid dimensions.")
    if width * height > max_pixels:
        raise InputImageError("The PNG image exceeds the configured pixel safety limit.")
    if int(getattr(image, "n_frames", 1)) != 1:
        raise InputImageError("The PNG image must contain exactly one frame.")


def decode_png_rgba(data: bytes, max_pixels: int) -> np.ndarray:
    """Decode one complete PNG frame into a copied uint8 RGBA array."""
    if not data:
        raise InputImageError("The PNG input is empty.")
    if max_pixels < 1:
        raise InputImageError("The PNG pixel safety limit must be positive.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                _check_png_header(probe, max_pixels)
                probe.verify()
            with Image.open(io.BytesIO(data)) as decoded:
                _check_png_header(decoded, max_pixels)
                decoded.load()
                return np.asarray(decoded.convert("RGBA"), dtype=np.uint8).copy()
    except InputImageError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        EOFError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise InputImageError("The input is not a complete, supported PNG image.") from exc


def validate_png_roundtrip(
    data: bytes,
    expected_rgba: np.ndarray,
    config: PixelizeConfig,
    allowed_palette: Sequence[RGB] = (),
) -> Dict[str, Any]:
    """Decode a PNG and add an exact decoded-pixel comparison to its report."""
    decoded = decode_png_rgba(data, config.width * config.height)
    report = validate_rgba(decoded, config, allowed_palette)
    checks = dict(report["checks"])
    metrics = dict(report["metrics"])

    try:
        expected = np.asarray(expected_rgba)
        shape_match = decoded.shape == expected.shape
        pixels_match = bool(shape_match and np.array_equal(decoded, expected))
        if shape_match:
            different_channels = decoded != expected
            different_channel_count = int(np.count_nonzero(different_channels))
            different_pixel_count = int(
                np.count_nonzero(np.any(different_channels, axis=2))
            )
        else:
            different_channel_count = None
            different_pixel_count = None
        metrics.update(
            {
                "roundtrip_expected_shape": list(expected.shape),
                "roundtrip_shape_match": shape_match,
                "roundtrip_different_channel_count": different_channel_count,
                "roundtrip_different_pixel_count": different_pixel_count,
            }
        )
    except Exception:
        pixels_match = False
        metrics.update(
            {
                "roundtrip_expected_shape": None,
                "roundtrip_shape_match": False,
                "roundtrip_different_channel_count": None,
                "roundtrip_different_pixel_count": None,
            }
        )

    checks["png_roundtrip"] = {
        "passed": pixels_match,
        "expected_shape": metrics["roundtrip_expected_shape"],
        "actual_shape": list(decoded.shape),
        "shape_match": metrics["roundtrip_shape_match"],
        "different_channel_count": metrics["roundtrip_different_channel_count"],
        "different_pixel_count": metrics["roundtrip_different_pixel_count"],
    }
    return {
        "passed": _all_checks_pass(checks),
        "checks": checks,
        "metrics": metrics,
    }
