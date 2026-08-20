import io
import json
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import server


def _png_chunk(chunk_type, payload):
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def in_memory_png():
    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixel_row = zlib.compress(b"\x00\xff\x00\x00\xff")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixel_row)
        + _png_chunk(b"IEND", b"")
    )


class ServerServiceTests(unittest.TestCase):
    def test_build_config_rejects_unknown_fields(self):
        config = server.build_config(
            {
                "width": 24,
                "height": 18,
                "max_colors": 12,
                "preview_scale": 4,
                "palette": ["#112233"],
            }
        )
        self.assertEqual((config.width, config.height), (24, 18))
        self.assertEqual(config.max_colors, 12)
        self.assertEqual(config.preview_scale, 4)
        self.assertEqual(config.max_input_bytes, server.MAX_FILE_BYTES)
        self.assertEqual(config.max_input_pixels, server.MAX_INPUT_PIXELS)

        with self.assertRaises(server.RequestError) as caught:
            server.build_config({"width": 24, "height": 18, "unknown": True})

        self.assertEqual(caught.exception.status, 422)
        self.assertEqual(caught.exception.error_type, "configuration")
        self.assertIn("unknown", caught.exception.message)

    def test_process_image_returns_fixed_artifacts_and_bundle(self):
        artifacts = {
            "sprite.png": b"sprite",
            "preview-4x.png": b"preview",
            "palette.png": b"palette",
            "palette.json": b'{"actual_color_count":3}',
            "report.json": b'{"passed":true}',
            "manifest.json": b'{"format":"true-pixelizer"}',
        }
        pixelizer_result = SimpleNamespace(
            artifact_bytes=lambda: artifacts,
            palette_data={"actual_color_count": 3},
            report={"passed": True},
            manifest={"format": "true-pixelizer"},
            warnings=("sample warning",),
            preview_filename="preview-4x.png",
        )

        with mock.patch.object(server, "pixelize_bytes", return_value=pixelizer_result) as pixelize:
            result = server.process_image(
                in_memory_png(),
                {"width": 16, "height": 12, "max_colors": 8, "preview_scale": 4},
            )

        self.assertEqual(set(result["artifacts"]), set(artifacts))
        self.assertEqual(result["preview_filename"], "preview-4x.png")
        self.assertEqual(result["summary"]["actual_color_count"], 3)
        self.assertTrue(result["summary"]["quality_passed"])
        self.assertEqual(result["warnings"], ["sample warning"])
        self.assertEqual(pixelize.call_args.args[0], in_memory_png())
        self.assertEqual(pixelize.call_args.args[1].width, 16)
        self.assertIsNone(pixelize.call_args.kwargs["mask_bytes"])

        with zipfile.ZipFile(io.BytesIO(result["bundle"])) as archive:
            self.assertEqual(set(archive.namelist()), set(artifacts))
            for name, payload in artifacts.items():
                self.assertEqual(archive.read(name), payload)

    def test_server_serves_health_page_and_upload_error(self):
        page = b"<!doctype html><title>True Pixelizer</title>"
        with tempfile.TemporaryDirectory() as temporary_directory:
            static_root = Path(temporary_directory)
            (static_root / "index.html").write_bytes(page)
            httpd = server.create_server(port=0, static_root=static_root)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            host, port = httpd.server_address[:2]
            base_url = "http://{}:{}".format(host, port)

            try:
                with urllib.request.urlopen(base_url + "/api/health", timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    health = json.loads(response.read().decode("utf-8"))
                self.assertTrue(health["ok"])
                self.assertEqual(health["service"], "true-pixelizer")

                with urllib.request.urlopen(base_url + "/", timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.read(), page)
                    self.assertEqual(response.headers.get_content_type(), "text/html")

                request = urllib.request.Request(
                    base_url + "/api/pixelize",
                    data=b"not-a-form-upload",
                    headers={"Content-Type": "application/octet-stream"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=3)
                error = caught.exception
                try:
                    self.assertEqual(error.code, 415)
                    payload = json.loads(error.read().decode("utf-8"))
                finally:
                    error.close()
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"]["type"], "request")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=3)

        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
