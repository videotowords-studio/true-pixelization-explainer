import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from true_pixelizer import PixelizeConfig, pixelize_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_source(path: Path) -> bytes:
    rgba = np.zeros((12, 10, 4), dtype=np.uint8)
    rgba[1:11, 1:9, :3] = (70, 180, 110)
    rgba[1:11, 1:9, 3] = 255
    rgba[4:8, 3:7, :3] = (240, 220, 80)
    output = io.BytesIO()
    Image.fromarray(rgba).save(output, format="PNG")
    payload = output.getvalue()
    path.write_bytes(payload)
    return payload


def write_oversized_png_header(path: Path) -> None:
    payload = struct.pack(">IIBBBBB", 20_000, 10_000, 8, 6, 0, 0, 0)
    chunk = b"IHDR" + payload
    ihdr = struct.pack(">I", len(payload)) + chunk + struct.pack(
        ">I", zlib.crc32(chunk) & 0xFFFFFFFF
    )
    iend_chunk = b"IEND"
    iend = struct.pack(">I", 0) + iend_chunk + struct.pack(
        ">I", zlib.crc32(iend_chunk) & 0xFFFFFFFF
    )
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + ihdr + iend)


class CliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        source_path = str(PROJECT_ROOT / "src")
        environment["PYTHONPATH"] = source_path + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
        return subprocess.run(
            [sys.executable, "-m", "true_pixelizer", *arguments],
            cwd=str(PROJECT_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_exports_json_and_matches_core_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.png"
            source_bytes = write_source(source_path)
            output_dir = root / "result"
            completed = self.run_cli(
                "pixelize",
                str(source_path),
                "--output-dir",
                str(output_dir),
                "--size",
                "8x8",
                "--max-colors",
                "3",
                "--alpha",
                "binary",
                "--preview-scale",
                "2",
                "--json",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["summary"]["quality_passed"])
            self.assertEqual(
                set(path.name for path in output_dir.iterdir()),
                {
                    "sprite.png",
                    "preview-2x.png",
                    "palette.png",
                    "palette.json",
                    "report.json",
                    "manifest.json",
                },
            )
            api_result = pixelize_bytes(
                source_bytes,
                PixelizeConfig(
                    width=8,
                    height=8,
                    max_colors=3,
                    alpha_mode="binary",
                    preview_scale=2,
                ),
            )
            self.assertEqual((output_dir / "sprite.png").read_bytes(), api_result.sprite_png)

    def test_nonempty_output_requires_force_and_preserves_unrelated_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            write_source(source)
            output = root / "result"
            output.mkdir()
            unrelated = output / "notes.txt"
            unrelated.write_text("keep", encoding="utf-8")
            arguments = (
                "pixelize",
                str(source),
                "--output-dir",
                str(output),
                "--size",
                "8x8",
                "--preview-scale",
                "2",
                "--json",
            )
            refused = self.run_cli(*arguments)
            self.assertEqual(refused.returncode, 2)
            self.assertFalse(json.loads(refused.stdout)["ok"])

            forced = self.run_cli(*arguments, "--force")
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")

    def test_validate_uses_exit_code_four_for_wrong_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            write_source(source)
            completed = self.run_cli(
                "validate",
                str(source),
                "--size",
                "9x9",
                "--max-colors",
                "8",
                "--alpha",
                "binary",
                "--json",
            )
            self.assertEqual(completed.returncode, 4)
            payload = json.loads(completed.stdout)
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["report"]["checks"]["dimensions"]["passed"])

    def test_missing_input_returns_exit_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = self.run_cli(
                "pixelize",
                str(Path(temporary) / "missing.png"),
                "--output-dir",
                str(Path(temporary) / "result"),
                "--size",
                "8x8",
                "--json",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(json.loads(completed.stdout)["ok"])

    def test_invalid_arguments_still_return_json(self) -> None:
        completed = self.run_cli(
            "pixelize",
            "missing.png",
            "--output-dir",
            "result",
            "--size",
            "bad",
            "--json",
        )
        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "arguments")
        self.assertEqual(completed.stderr, "")

    def test_force_refuses_to_overwrite_source_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sprite.png"
            original = write_source(source)
            completed = self.run_cli(
                "pixelize",
                str(source),
                "--output-dir",
                str(root),
                "--size",
                "8x8",
                "--preview-scale",
                "2",
                "--force",
                "--json",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                json.loads(completed.stdout)["error"]["type"], "configuration"
            )

    def test_palette_file_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            write_source(source)
            palette = root / "fixed-palette.json"
            palette.write_text(
                json.dumps({"colors": ["#000000", "#FFFFFF"]}),
                encoding="utf-8",
            )
            output = root / "result"
            completed = self.run_cli(
                "pixelize",
                str(source),
                "--output-dir",
                str(output),
                "--size",
                "8x8",
                "--max-colors",
                "2",
                "--palette-file",
                str(palette),
                "--preview-scale",
                "2",
                "--json",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            palette_data = json.loads((output / "palette.json").read_text())
            self.assertEqual(palette_data["mode"], "fixed")
            self.assertEqual(
                [entry["hex"] for entry in palette_data["generated_colors"]],
                ["#000000", "#FFFFFF"],
            )

    def test_filesystem_alias_to_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            original = write_source(source)
            output = root / "result"
            output.mkdir()
            alias = output / "sprite.png"
            os.link(source, alias)
            completed = self.run_cli(
                "pixelize",
                str(source),
                "--output-dir",
                str(output),
                "--size",
                "8x8",
                "--force",
                "--json",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                json.loads(completed.stdout)["error"]["type"], "configuration"
            )

    def test_case_insensitive_source_alias_is_rejected_when_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Sprite.png"
            original = write_source(source)
            if not (root / "sprite.png").exists():
                self.skipTest("The temporary filesystem is case-sensitive.")
            completed = self.run_cli(
                "pixelize",
                str(source),
                "--output-dir",
                str(root),
                "--size",
                "8x8",
                "--force",
                "--json",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                json.loads(completed.stdout)["error"]["type"], "configuration"
            )

    def test_decompression_bomb_returns_input_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "oversized.png"
            write_oversized_png_header(source)
            completed = self.run_cli(
                "pixelize",
                str(source),
                "--output-dir",
                str(root / "result"),
                "--size",
                "1x1",
                "--preview-scale",
                "1",
                "--json",
            )
            self.assertEqual(completed.returncode, 2)
            payload = json.loads(completed.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "input")


if __name__ == "__main__":
    unittest.main()
