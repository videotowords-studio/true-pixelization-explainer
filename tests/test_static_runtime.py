import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "pyodide-0.29.4"
EXPECTED_SHA256 = {
    "numpy-2.2.5-cp313-cp313-pyemscripten_2025_0_wasm32.whl": (
        "800c98edc0c864dfa49f07005680c699b4b42b84eae1f8cb19d35b3634e7f05c"
    ),
    "pillow-11.3.0-cp313-cp313-pyemscripten_2025_0_wasm32.whl": (
        "57d88e2ac283c21830b4ee920cb73fbbf5c46df62d967089fce5fec46548bd7b"
    ),
    "pyodide-lock.json": (
        "14d2c2dba101277999e17135e653d8f15389ad1437f53eae213bf0c3cdff723d"
    ),
    "pyodide.asm.js": (
        "fe75e97ef2c7a10c41f23b96344c553c7c1c62821bb206080dbb24941c06d5f3"
    ),
    "pyodide.asm.wasm": (
        "10090fe41e019ae669d512e1f747021a8db2aaab0f6dd6f85fa9368c55d681e3"
    ),
    "pyodide.js": (
        "7be9e63eddf6bd8786a5ab71c8689c35731d187e3415fe58d7cb95423b50421a"
    ),
    "python_stdlib.zip": (
        "92cb24faa546818f3ef4050fd5bd2b6487bd2042efed2113af141d035f30efb4"
    ),
}


class StaticRuntimeTests(unittest.TestCase):
    def test_runtime_assets_match_pinned_release(self):
        for filename, expected in EXPECTED_SHA256.items():
            payload = (RUNTIME_ROOT / filename).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected, filename)

        lock = json.loads((RUNTIME_ROOT / "pyodide-lock.json").read_text())
        self.assertEqual(lock["packages"]["numpy"]["version"], "2.2.5")
        self.assertEqual(lock["packages"]["pillow"]["version"], "11.3.0")

    def test_worker_loads_runtime_from_the_site(self):
        worker = (PROJECT_ROOT / "pixelizer-worker.js").read_text()
        self.assertIn("new URL('vendor/pyodide-0.29.4/', PROJECT_URL).href", worker)
        self.assertNotIn("cdn.jsdelivr.net", worker)


if __name__ == "__main__":
    unittest.main()
