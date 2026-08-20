import hashlib
import io
import json
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, __version__ as pillow_version

from .colors import RGB, rgb_to_hex
from .config import PixelizeConfig
from .errors import InputImageError, PixelizerError, ProcessingError, QualityError
from .exporter import describe_artifacts, json_bytes
from .image_ops import load_rgba_and_mask, reconstruct_grid
from .palette import build_palette, map_grid_to_palette
from .quality import decode_png_rgba, validate_png_roundtrip
from .version import ALGORITHM_VERSION, PACKAGE_VERSION


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rgba_sha256(rgba: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(rgba, dtype=np.uint8).tobytes())


def _encode_rgba_png(rgba: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return output.getvalue()


def _make_preview(rgba: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(rgba, scale, axis=0), scale, axis=1)


def _make_palette_image(palette: Sequence[RGB]) -> np.ndarray:
    cell_size = 32
    columns = min(16, max(1, len(palette)))
    rows = (len(palette) + columns - 1) // columns
    image = np.zeros((rows * cell_size, columns * cell_size, 4), dtype=np.uint8)
    image[..., 3] = 255
    for index, color in enumerate(palette):
        row, column = divmod(index, columns)
        image[
            row * cell_size : (row + 1) * cell_size,
            column * cell_size : (column + 1) * cell_size,
            :3,
        ] = color
    return image


def _used_palette_data(
    rgba: np.ndarray,
    generated_palette: Sequence[RGB],
    config: PixelizeConfig,
) -> Dict[str, Any]:
    visible = rgba[..., 3] > 0
    colors, counts = np.unique(rgba[..., :3][visible], axis=0, return_counts=True)
    count_by_color = {
        tuple(int(channel) for channel in color): int(count)
        for color, count in zip(colors, counts)
    }
    generated = []
    for color in generated_palette:
        generated.append(
            {
                "hex": rgb_to_hex(color),
                "rgb": list(color),
                "visible_pixels": count_by_color.get(tuple(color), 0),
            }
        )
    used = [entry for entry in generated if entry["visible_pixels"] > 0]
    return {
        "schema_version": "1.0",
        "maximum_colors": config.max_colors,
        "actual_color_count": len(colors),
        "mode": "fixed" if config.palette else "automatic",
        "generated_colors": generated,
        "used_colors": used,
        "locked_colors": [rgb_to_hex(color) for color in config.locked_colors],
    }


def _quantization_metrics(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
    visible = after[..., 3] > 0
    before_rgb = before[..., :3][visible].astype(np.float64)
    after_rgb = after[..., :3][visible].astype(np.float64)
    changed = np.any(before_rgb != after_rgb, axis=1)
    alpha_weights = after[..., 3][visible].astype(np.float64) / 255.0
    squared_error = np.sum((before_rgb - after_rgb) ** 2, axis=1)
    if float(np.sum(alpha_weights)) > 0.0:
        weighted_rmse = float(
            np.sqrt(np.average(squared_error / 3.0, weights=alpha_weights))
        )
    else:
        weighted_rmse = 0.0
    return {
        "changed_visible_pixels": int(np.count_nonzero(changed)),
        "changed_visible_ratio": float(np.mean(changed)) if len(changed) else 0.0,
        "alpha_weighted_rgb_rmse": weighted_rmse,
    }


@dataclass(frozen=True)
class PixelizeResult:
    sprite_png: bytes
    preview_png: bytes
    palette_png: bytes
    palette: Tuple[RGB, ...]
    _palette_data_json: bytes = field(repr=False)
    _report_json: bytes = field(repr=False)
    _manifest_json: bytes = field(repr=False)
    warnings: Tuple[str, ...]
    preview_scale: int

    @property
    def palette_data(self) -> Dict[str, Any]:
        return json.loads(self._palette_data_json.decode("utf-8"))

    @property
    def report(self) -> Dict[str, Any]:
        return json.loads(self._report_json.decode("utf-8"))

    @property
    def manifest(self) -> Dict[str, Any]:
        return json.loads(self._manifest_json.decode("utf-8"))

    @property
    def preview_filename(self) -> str:
        return "preview-{}x.png".format(self.preview_scale)

    def artifact_bytes(self) -> Dict[str, bytes]:
        return {
            "sprite.png": self.sprite_png,
            self.preview_filename: self.preview_png,
            "palette.png": self.palette_png,
            "palette.json": self._palette_data_json,
            "report.json": self._report_json,
            "manifest.json": self._manifest_json,
        }


def pixelize_bytes(
    image_bytes: bytes,
    config: PixelizeConfig,
    mask_bytes: Optional[bytes] = None,
) -> PixelizeResult:
    if len(image_bytes) > config.max_input_bytes:
        raise InputImageError(
            "The encoded input image exceeds the configured byte safety limit."
        )
    if mask_bytes is not None and len(mask_bytes) > config.max_input_bytes:
        raise InputImageError(
            "The encoded mask exceeds the configured byte safety limit."
        )
    try:
        return _pixelize_bytes_impl(image_bytes, config, mask_bytes)
    except PixelizerError:
        raise
    except MemoryError as exc:
        raise ProcessingError(
            "The image could not be processed within the available memory."
        ) from exc
    except (
        FloatingPointError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ProcessingError(
            "The deterministic image processing pipeline could not finish."
        ) from exc


def _pixelize_bytes_impl(
    image_bytes: bytes,
    config: PixelizeConfig,
    mask_bytes: Optional[bytes],
) -> PixelizeResult:
    subject, input_metadata, input_warnings = load_rgba_and_mask(
        image_bytes, config, mask_bytes
    )
    sampled_grid, grid_metadata, grid_warnings = reconstruct_grid(subject, config)

    visible = sampled_grid[..., 3] > 0
    palette = build_palette(
        sampled_grid[..., :3][visible],
        config.max_colors,
        config.palette,
        config.locked_colors,
    )
    final_grid = map_grid_to_palette(sampled_grid, palette, config.dither)
    final_grid[final_grid[..., 3] == 0, :3] = 0

    sprite_png = _encode_rgba_png(final_grid)
    allowed_palette = config.palette if config.palette else ()
    quality = validate_png_roundtrip(
        sprite_png,
        final_grid,
        config,
        allowed_palette=allowed_palette,
    )

    preview_grid = _make_preview(final_grid, config.preview_scale)
    preview_png = _encode_rgba_png(preview_grid)
    decoded_preview = decode_png_rgba(
        preview_png,
        config.width
        * config.height
        * config.preview_scale
        * config.preview_scale,
    )
    preview_matches = bool(np.array_equal(decoded_preview, preview_grid))
    quality["checks"]["nearest_neighbor_preview"] = {
        "passed": preview_matches,
        "expected_size": [
            config.width * config.preview_scale,
            config.height * config.preview_scale,
        ],
        "actual_size": [int(decoded_preview.shape[1]), int(decoded_preview.shape[0])],
        "integer_scale": config.preview_scale,
    }
    quality["passed"] = bool(quality["passed"] and preview_matches)

    warnings = tuple(input_warnings + grid_warnings)
    quality["warnings"] = list(warnings)
    quality["processing"] = {
        "input": input_metadata,
        "grid": grid_metadata,
        "quantization": _quantization_metrics(sampled_grid, final_grid),
    }
    if not quality["passed"]:
        failed_checks = [
            name
            for name, value in quality["checks"].items()
            if not value.get("passed", False)
        ]
        raise QualityError(
            "Hard output validation failed: {}.".format(", ".join(failed_checks))
        )

    palette_data = _used_palette_data(final_grid, palette, config)
    palette_png = _encode_rgba_png(_make_palette_image(palette))
    manifest = {
        "schema_version": "1.0",
        "generator": {
            "name": "true-pixelizer",
            "version": PACKAGE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "python_version": platform.python_version(),
            "pillow_version": pillow_version,
            "numpy_version": np.__version__,
        },
        "config": config.to_dict(),
        "inputs": {
            "image_sha256": _sha256(image_bytes),
            "mask_sha256": _sha256(mask_bytes) if mask_bytes is not None else None,
        },
        "pixel_hashes": {
            "sprite_rgba_sha256": _rgba_sha256(final_grid),
            "preview_rgba_sha256": _rgba_sha256(preview_grid),
        },
        "outputs": {
            "sprite": "sprite.png",
            "preview": "preview-{}x.png".format(config.preview_scale),
            "palette": "palette.png",
            "palette_data": "palette.json",
            "quality_report": "report.json",
        },
        "warnings": list(warnings),
    }

    palette_data_json = json_bytes(palette_data)
    report_json = json_bytes(quality)
    base_artifacts = {
        "sprite.png": sprite_png,
        "preview-{}x.png".format(config.preview_scale): preview_png,
        "palette.png": palette_png,
        "palette.json": palette_data_json,
        "report.json": report_json,
    }
    manifest["artifacts"] = describe_artifacts(base_artifacts)
    manifest_json = json_bytes(manifest)

    return PixelizeResult(
        sprite_png=sprite_png,
        preview_png=preview_png,
        palette_png=palette_png,
        palette=palette,
        _palette_data_json=palette_data_json,
        _report_json=report_json,
        _manifest_json=manifest_json,
        warnings=warnings,
        preview_scale=config.preview_scale,
    )
