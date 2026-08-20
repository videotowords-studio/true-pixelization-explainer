import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from true_pixelizer.errors import ExportError
from true_pixelizer.exporter import write_artifacts


class ExporterTests(unittest.TestCase):
    def test_force_failure_rolls_back_every_matching_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            old_artifacts = {
                "palette.png": b"old-palette",
                "sprite.png": b"old-sprite",
                "manifest.json": b"old-manifest",
            }
            for name, payload in old_artifacts.items():
                (output / name).write_bytes(payload)

            real_replace = os.replace
            failure_injected = False

            def flaky_replace(source: str, target: str) -> None:
                nonlocal failure_injected
                source_path = Path(source)
                target_path = Path(target)
                is_staged_sprite = (
                    target_path == output / "sprite.png"
                    and source_path.name == "sprite.png"
                    and source_path.parent.name.startswith(".result.tmp-")
                )
                if is_staged_sprite and not failure_injected:
                    failure_injected = True
                    raise OSError("injected replacement failure")
                real_replace(source, target)

            with mock.patch(
                "true_pixelizer.exporter.os.replace", side_effect=flaky_replace
            ):
                with self.assertRaises(ExportError):
                    write_artifacts(
                        output,
                        {
                            "palette.png": b"new-palette",
                            "sprite.png": b"new-sprite",
                            "manifest.json": b"new-manifest",
                        },
                        force=True,
                    )

            self.assertTrue(failure_injected)
            self.assertEqual(
                {name: (output / name).read_bytes() for name in old_artifacts},
                old_artifacts,
            )


if __name__ == "__main__":
    unittest.main()
