'use strict';

const PROJECT_URL = new URL('./', self.location.href);
const PYODIDE_INDEX_URL = new URL('vendor/pyodide-0.29.4/', PROJECT_URL).href;
const PYTHON_ROOT = '/home/pyodide';
const SOURCE_FILES = [
  'server.py',
  'true-pixelizer/src/true_pixelizer/__init__.py',
  'true-pixelizer/src/true_pixelizer/__main__.py',
  'true-pixelizer/src/true_pixelizer/cli.py',
  'true-pixelizer/src/true_pixelizer/colors.py',
  'true-pixelizer/src/true_pixelizer/config.py',
  'true-pixelizer/src/true_pixelizer/errors.py',
  'true-pixelizer/src/true_pixelizer/exporter.py',
  'true-pixelizer/src/true_pixelizer/image_ops.py',
  'true-pixelizer/src/true_pixelizer/palette.py',
  'true-pixelizer/src/true_pixelizer/pipeline.py',
  'true-pixelizer/src/true_pixelizer/quality.py',
  'true-pixelizer/src/true_pixelizer/version.py'
];

let enginePromise = null;

function postStatus(stage) {
  self.postMessage({ type: 'status', stage });
}

async function fetchSource(relativePath) {
  const response = await fetch(new URL(relativePath, PROJECT_URL), { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`无法载入真像素化程序文件：${relativePath}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function installSource(pyodide) {
  const entries = await Promise.all(
    SOURCE_FILES.map(async relativePath => [relativePath, await fetchSource(relativePath)])
  );
  entries.forEach(([relativePath, bytes]) => {
    const target = `${PYTHON_ROOT}/${relativePath}`;
    const directory = target.slice(0, target.lastIndexOf('/'));
    pyodide.FS.mkdirTree(directory);
    pyodide.FS.writeFile(target, bytes);
  });
}

async function initializeEngine() {
  if (enginePromise) return enginePromise;
  enginePromise = (async () => {
    postStatus('runtime');
    importScripts(`${PYODIDE_INDEX_URL}pyodide.js`);
    const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });

    postStatus('packages');
    await pyodide.loadPackage(['numpy', 'pillow']);

    postStatus('source');
    await installSource(pyodide);
    await pyodide.runPythonAsync(`
import json
from pathlib import Path

from server import RequestError, process_image

_TP_RESULT_ROOT = Path('/tmp/true-pixelizer-result')


def _tp_browser_process(config_json, image_path, mask_path):
    try:
        image_bytes = Path(image_path).read_bytes()
        mask_bytes = Path(mask_path).read_bytes() if mask_path else None
        result = process_image(image_bytes, json.loads(config_json), mask_bytes=mask_bytes)
    except RequestError as exc:
        return json.dumps({
            'ok': False,
            'error': {'type': exc.error_type, 'message': exc.message},
        }, ensure_ascii=False)
    except Exception:
        return json.dumps({
            'ok': False,
            'error': {'type': 'internal', 'message': '浏览器处理引擎未能完成任务，请缩小图片后重试。'},
        }, ensure_ascii=False)

    _TP_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_paths = {}
    for name, payload in result['artifacts'].items():
        path = _TP_RESULT_ROOT / name
        path.write_bytes(payload)
        artifact_paths[name] = str(path)

    bundle_path = _TP_RESULT_ROOT / 'true-pixelizer-result.zip'
    bundle_path.write_bytes(result['bundle'])
    payload = {
        'ok': True,
        'summary': result['summary'],
        'warnings': result['warnings'],
        'palette_data': result['palette_data'],
        'report': result['report'],
        'manifest': result['manifest'],
    }
    files = {
        'sprite': {
            'path': artifact_paths['sprite.png'],
            'name': 'sprite.png',
            'type': 'image/png',
        },
        'preview': {
            'path': artifact_paths[result['preview_filename']],
            'name': result['preview_filename'],
            'type': 'image/png',
        },
        'report': {
            'path': artifact_paths['report.json'],
            'name': 'report.json',
            'type': 'application/json',
        },
        'manifest': {
            'path': artifact_paths['manifest.json'],
            'name': 'manifest.json',
            'type': 'application/json',
        },
        'bundle': {
            'path': str(bundle_path),
            'name': 'true-pixelizer-result.zip',
            'type': 'application/zip',
        },
    }
    return json.dumps({'ok': True, 'payload': payload, 'files': files}, ensure_ascii=False)
`);
    return pyodide;
  })();
  return enginePromise;
}

function readResultFiles(pyodide, descriptors) {
  const files = {};
  const transfers = [];
  Object.entries(descriptors).forEach(([key, descriptor]) => {
    const bytes = pyodide.FS.readFile(descriptor.path).slice();
    files[key] = {
      name: descriptor.name,
      type: descriptor.type,
      buffer: bytes.buffer
    };
    transfers.push(bytes.buffer);
  });
  return { files, transfers };
}

async function handleProcess(message) {
  const pyodide = await initializeEngine();
  const imagePath = '/tmp/true-pixelizer-source.png';
  const maskPath = message.mask ? '/tmp/true-pixelizer-mask.png' : '';
  pyodide.FS.writeFile(imagePath, new Uint8Array(message.image));
  if (message.mask) pyodide.FS.writeFile(maskPath, new Uint8Array(message.mask));

  pyodide.globals.set('_tp_config_json', JSON.stringify(message.config));
  pyodide.globals.set('_tp_image_path', imagePath);
  pyodide.globals.set('_tp_mask_path', maskPath);
  const rawResult = await pyodide.runPythonAsync(
    '_tp_browser_process(_tp_config_json, _tp_image_path, _tp_mask_path)'
  );
  const result = JSON.parse(rawResult);
  if (!result.ok) {
    self.postMessage({
      type: 'request-error',
      requestId: message.requestId,
      error: result.error
    });
    return;
  }

  const { files, transfers } = readResultFiles(pyodide, result.files);
  self.postMessage({
    type: 'result',
    requestId: message.requestId,
    payload: result.payload,
    files
  }, transfers);
}

self.addEventListener('message', event => {
  const message = event.data || {};
  if (message.type === 'init') {
    initializeEngine()
      .then(() => self.postMessage({ type: 'ready' }))
      .catch(error => {
        enginePromise = null;
        self.postMessage({ type: 'fatal', message: error.message || '浏览器处理引擎加载失败。' });
      });
    return;
  }
  if (message.type === 'process') {
    handleProcess(message).catch(error => {
      self.postMessage({
        type: 'request-error',
        requestId: message.requestId,
        error: { type: 'internal', message: error.message || '浏览器处理失败。' }
      });
    });
  }
});
