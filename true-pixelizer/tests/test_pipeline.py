import io
import json
import struct
import unittest
import zlib

import numpy as np
from PIL import Image

from true_pixelizer import PixelizeConfig, pixelize_bytes
from true_pixelizer.errors import ConfigurationError, InputImageError
from true_pixelizer.quality import decode_png_rgba, validate_rgba


def png_bytes(rgba: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8)).save(output, format="PNG")
    return output.getvalue()


def oversized_png_header(width: int, height: int) -> bytes:
    payload = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    chunk = b"IHDR" + payload
    ihdr = struct.pack(">I", len(payload)) + chunk + struct.pack(
        ">I", zlib.crc32(chunk) & 0xFFFFFFFF
    )
    iend_chunk = b"IEND"
    iend = struct.pack(">I", 0) + iend_chunk + struct.pack(
        ">I", zlib.crc32(iend_chunk) & 0xFFFFFFFF
    )
    return b"\x89PNG\r\n\x1a\n" + ihdr + iend


class PipelineTests(unittest.TestCase):
    def test_binary_contract_and_repeatability(self) -> None:
        source = np.zeros((18, 14, 4), dtype=np.uint8)
        for y in range(2, 16):
            for x in range(2, 12):
                source[y, x] = (
                    x * 19 % 256,
                    y * 17 % 256,
                    (x + y) * 13 % 256,
                    255 if (x + y) % 3 else 190,
                )
        config = PixelizeConfig(
            width=8,
            height=8,
            max_colors=4,
            alpha_mode="binary",
            preview_scale=4,
        )

        first = pixelize_bytes(png_bytes(source), config)
        second = pixelize_bytes(png_bytes(source), config)

        self.assertEqual(first.sprite_png, second.sprite_png)
        self.assertEqual(first.preview_png, second.preview_png)
        self.assertTrue(first.report["passed"])
        sprite = decode_png_rgba(first.sprite_png, 64)
        self.assertEqual(sprite.shape, (8, 8, 4))
        self.assertLessEqual(
            len(np.unique(sprite[..., :3][sprite[..., 3] > 0], axis=0)), 4
        )
        self.assertTrue(set(np.unique(sprite[..., 3])).issubset({0, 255}))
        self.assertTrue(np.all(sprite[..., :3][sprite[..., 3] == 0] == 0))

    def test_fixed_palette_is_enforced(self) -> None:
        source = np.asarray(
            [
                [(255, 0, 0, 255), (0, 255, 0, 255)],
                [(0, 0, 255, 255), (240, 240, 240, 255)],
            ],
            dtype=np.uint8,
        )
        palette = ((0, 0, 0), (255, 255, 255))
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=2,
                height=2,
                max_colors=2,
                palette=palette,
                alpha_mode="preserve",
                preview_scale=2,
            ),
        )
        sprite = decode_png_rgba(result.sprite_png, 4)
        used = {
            tuple(int(channel) for channel in color)
            for color in np.unique(sprite[..., :3], axis=0).reshape((-1, 3))
        }
        self.assertTrue(used.issubset(set(palette)))
        self.assertEqual(result.palette, palette)

    def test_premultiplied_area_sampling_avoids_hidden_rgb_bleed(self) -> None:
        source = np.asarray(
            [[(255, 0, 0, 0), (0, 0, 255, 255)]], dtype=np.uint8
        )
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=1,
                height=1,
                max_colors=8,
                alpha_mode="preserve",
                trim="none",
                preview_scale=1,
            ),
        )
        pixel = decode_png_rgba(result.sprite_png, 1)[0, 0]
        self.assertEqual(tuple(pixel[:3]), (0, 0, 255))
        self.assertEqual(int(pixel[3]), 128)

    def test_binary_alpha_threshold_uses_half_as_inclusive(self) -> None:
        source = np.asarray(
            [[(255, 0, 0, 127), (0, 0, 255, 128)]], dtype=np.uint8
        )
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=2,
                height=1,
                max_colors=2,
                alpha_mode="binary",
                alpha_threshold=0.5,
                trim="none",
                preview_scale=1,
            ),
        )
        alpha = decode_png_rgba(result.sprite_png, 2)[0, :, 3]
        self.assertEqual(alpha.tolist(), [0, 255])

    def test_key_color_removes_background(self) -> None:
        source = np.asarray(
            [[(255, 0, 255, 255), (220, 30, 40, 255)]], dtype=np.uint8
        )
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=2,
                height=1,
                max_colors=2,
                key_color=(255, 0, 255),
                trim="none",
                preview_scale=1,
            ),
        )
        sprite = decode_png_rgba(result.sprite_png, 2)
        self.assertEqual(sprite[0, :, 3].tolist(), [0, 255])
        self.assertEqual(sprite[0, 0, :3].tolist(), [0, 0, 0])

    def test_explicit_mask_controls_alpha(self) -> None:
        source = np.full((2, 2, 4), (30, 180, 90, 255), dtype=np.uint8)
        mask = np.asarray([[0, 255], [0, 255]], dtype=np.uint8)
        mask_output = io.BytesIO()
        Image.fromarray(mask).save(mask_output, format="PNG")
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=2,
                height=2,
                max_colors=2,
                trim="none",
                preview_scale=1,
            ),
            mask_bytes=mask_output.getvalue(),
        )
        sprite = decode_png_rgba(result.sprite_png, 4)
        self.assertEqual(sprite[..., 3].tolist(), [[0, 255], [0, 255]])

    def test_mask_size_mismatch_fails(self) -> None:
        source = np.full((2, 2, 4), 255, dtype=np.uint8)
        mask_output = io.BytesIO()
        Image.fromarray(np.full((1, 1), 255, dtype=np.uint8)).save(
            mask_output, format="PNG"
        )
        with self.assertRaises(InputImageError):
            pixelize_bytes(
                png_bytes(source),
                PixelizeConfig(width=2, height=2, preview_scale=1),
                mask_bytes=mask_output.getvalue(),
            )

    def test_fully_transparent_input_fails(self) -> None:
        source = np.zeros((2, 2, 4), dtype=np.uint8)
        with self.assertRaises(InputImageError):
            pixelize_bytes(
                png_bytes(source),
                PixelizeConfig(width=2, height=2, preview_scale=1),
            )

    def test_artifact_bundle_contains_fixed_outputs(self) -> None:
        source = np.full((2, 2, 4), (20, 40, 60, 255), dtype=np.uint8)
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(width=2, height=2, preview_scale=3),
        )
        self.assertEqual(
            set(result.artifact_bytes()),
            {
                "sprite.png",
                "preview-3x.png",
                "palette.png",
                "palette.json",
                "report.json",
                "manifest.json",
            },
        )

    def test_one_dimensional_source_uses_nearest_neighbor_when_enlarged(self) -> None:
        source = np.asarray(
            [[(255, 0, 0, 255), (0, 0, 255, 255)]], dtype=np.uint8
        )
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=4,
                height=2,
                max_colors=2,
                alpha_mode="preserve",
                preview_scale=1,
            ),
        )
        sprite = decode_png_rgba(result.sprite_png, 8)
        expected_row = [
            [255, 0, 0, 255],
            [255, 0, 0, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ]
        self.assertEqual(sprite[0].tolist(), expected_row)
        self.assertEqual(sprite[1].tolist(), expected_row)

    def test_locked_color_is_first_and_uses_a_palette_slot(self) -> None:
        source = np.asarray(
            [[(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]],
            dtype=np.uint8,
        )
        locked = (255, 0, 255)
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=3,
                height=1,
                max_colors=2,
                locked_colors=(locked,),
                preview_scale=1,
            ),
        )
        self.assertEqual(result.palette[0], locked)
        self.assertLessEqual(len(result.palette), 2)

    def test_bayer_dither_is_repeatable(self) -> None:
        source = np.zeros((8, 8, 4), dtype=np.uint8)
        source[..., 3] = 255
        for y in range(8):
            for x in range(8):
                source[y, x, :3] = (x * 31, y * 31, (x + y) * 15)
        config = PixelizeConfig(
            width=8,
            height=8,
            max_colors=2,
            dither="bayer4",
            preview_scale=1,
        )
        first = pixelize_bytes(png_bytes(source), config)
        second = pixelize_bytes(png_bytes(source), config)
        self.assertEqual(first.sprite_png, second.sprite_png)

    def test_animated_input_is_rejected(self) -> None:
        first = Image.fromarray(np.full((2, 2, 3), 20, dtype=np.uint8))
        second = Image.fromarray(np.full((2, 2, 3), 220, dtype=np.uint8))
        animated = io.BytesIO()
        first.save(
            animated,
            format="GIF",
            save_all=True,
            append_images=[second],
            duration=100,
            loop=0,
        )
        with self.assertRaises(InputImageError):
            pixelize_bytes(
                animated.getvalue(),
                PixelizeConfig(width=2, height=2, preview_scale=1),
            )

    def test_preview_memory_limit_is_validated_before_processing(self) -> None:
        with self.assertRaises(ConfigurationError):
            PixelizeConfig(width=1024, height=1024, preview_scale=8)

    def test_mixed_axis_resize_keeps_nearest_neighbor_on_enlarged_axis(self) -> None:
        row = np.asarray(
            [[(255, 0, 0, 255), (0, 0, 255, 255)]], dtype=np.uint8
        )
        source = np.repeat(row, 4, axis=0)
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=3,
                height=1,
                max_colors=3,
                alpha_mode="preserve",
                fit="stretch",
                trim="none",
                preview_scale=1,
            ),
        )
        sprite = decode_png_rgba(result.sprite_png, 3)
        self.assertEqual(
            sprite[0].tolist(),
            [
                [255, 0, 0, 255],
                [0, 0, 255, 255],
                [0, 0, 255, 255],
            ],
        )

    def test_output_roundtrip_does_not_use_source_pixel_limit(self) -> None:
        source = np.asarray([[(20, 40, 60, 255)]], dtype=np.uint8)
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(
                width=2,
                height=2,
                max_input_pixels=1,
                preview_scale=1,
            ),
        )
        self.assertTrue(result.report["passed"])

    def test_quality_rejects_non_uint8_rgba(self) -> None:
        invalid = np.asarray([[[999.0, -1.0, np.nan, 300.0]]])
        report = validate_rgba(
            invalid,
            PixelizeConfig(
                width=1,
                height=1,
                alpha_mode="preserve",
                preview_scale=1,
            ),
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["pixel_format"]["passed"])

    def test_result_metadata_is_defensive_and_matches_export(self) -> None:
        source = np.asarray([[(20, 40, 60, 255)]], dtype=np.uint8)
        result = pixelize_bytes(
            png_bytes(source),
            PixelizeConfig(width=1, height=1, preview_scale=1),
        )
        artifacts_before = result.artifact_bytes()
        exported_manifest = json.loads(
            artifacts_before["manifest.json"].decode("utf-8")
        )
        self.assertEqual(result.manifest, exported_manifest)
        mutated_report = result.report
        mutated_report["passed"] = False
        mutated_manifest = result.manifest
        mutated_manifest["artifacts"] = {}
        self.assertTrue(result.report["passed"])
        self.assertEqual(result.manifest, exported_manifest)
        self.assertEqual(result.artifact_bytes(), artifacts_before)

    def test_encoded_input_byte_limit_is_enforced_by_core_api(self) -> None:
        source = np.asarray([[(20, 40, 60, 255)]], dtype=np.uint8)
        with self.assertRaises(InputImageError):
            pixelize_bytes(
                png_bytes(source),
                PixelizeConfig(
                    width=1,
                    height=1,
                    preview_scale=1,
                    max_input_bytes=1,
                ),
            )

    def test_decompression_bomb_is_classified_as_input_error(self) -> None:
        with self.assertRaises(InputImageError):
            pixelize_bytes(
                oversized_png_header(20_000, 10_000),
                PixelizeConfig(width=1, height=1, preview_scale=1),
            )


if __name__ == "__main__":
    unittest.main()
