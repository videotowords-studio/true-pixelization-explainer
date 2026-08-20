# Browser runtime

This directory contains the browser runtime required by the static deployment:

- Pyodide 0.29.4
- NumPy 2.2.5
- Pillow 11.3.0

The files are mirrored from `https://cdn.jsdelivr.net/pyodide/v0.29.4/full/` so the published tool does not depend on a third-party runtime CDN at page load. Package hashes remain recorded in `pyodide-lock.json`.
