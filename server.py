#!/usr/bin/env python3
"""Serve the page and expose the local true-pixelizer core over HTTP."""

import argparse
import io
import json
import secrets
import sys
import threading
import traceback
import zipfile
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent
PIXELIZER_SRC = PROJECT_ROOT / "true-pixelizer" / "src"
if str(PIXELIZER_SRC) not in sys.path:
    sys.path.insert(0, str(PIXELIZER_SRC))

from true_pixelizer import PixelizeConfig, pixelize_bytes  # noqa: E402
from true_pixelizer.colors import RGB, parse_hex_color  # noqa: E402
from true_pixelizer.errors import (  # noqa: E402
    ConfigurationError,
    InputImageError,
    PixelizerError,
    ProcessingError,
    QualityError,
)
from true_pixelizer.version import PACKAGE_VERSION  # noqa: E402


MAX_FILE_BYTES = 20_000_000
MAX_REQUEST_BYTES = MAX_FILE_BYTES * 2 + 2_000_000
MAX_INPUT_PIXELS = 16_000_000
PROCESSING_SLOTS = threading.BoundedSemaphore(2)

CONFIG_FIELDS = {
    "width",
    "height",
    "max_colors",
    "alpha_mode",
    "alpha_threshold",
    "fit",
    "anchor",
    "trim",
    "padding",
    "dither",
    "preview_scale",
    "palette",
    "locked_colors",
    "key_color",
    "key_tolerance",
}


class RequestError(Exception):
    def __init__(self, status: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message


class UploadedPart:
    def __init__(
        self,
        name: str,
        data: bytes,
        filename: Optional[str] = None,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.name = name
        self.data = data
        self.filename = filename
        self.content_type = content_type


def _integer(raw: Mapping[str, Any], name: str, default: Optional[int] = None) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(422, "configuration", "参数 {} 必须是整数。".format(name))
    return value


def _number(
    raw: Mapping[str, Any], name: str, default: Optional[float] = None
) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(422, "configuration", "参数 {} 必须是数字。".format(name))
    return float(value)


def _choice(
    raw: Mapping[str, Any], name: str, allowed: Sequence[str], default: str
) -> str:
    value = raw.get(name, default)
    if not isinstance(value, str) or value not in allowed:
        raise RequestError(422, "configuration", "参数 {} 的取值无效。".format(name))
    return value


def _colors(raw: Mapping[str, Any], name: str) -> Tuple[RGB, ...]:
    value = raw.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RequestError(422, "configuration", "参数 {} 必须是颜色列表。".format(name))
    try:
        return tuple(parse_hex_color(item) for item in value)
    except ConfigurationError as exc:
        raise RequestError(422, "configuration", "{} 中包含无效颜色。".format(name)) from exc


def build_config(raw: Mapping[str, Any]) -> PixelizeConfig:
    if not isinstance(raw, dict):
        raise RequestError(400, "request", "config 必须是 JSON 对象。")
    unexpected = sorted(set(raw) - CONFIG_FIELDS)
    if unexpected:
        raise RequestError(
            422,
            "configuration",
            "存在不支持的参数：{}。".format("、".join(unexpected)),
        )

    key_color_value = raw.get("key_color")
    if key_color_value in (None, ""):
        key_color = None
    elif isinstance(key_color_value, str):
        try:
            key_color = parse_hex_color(key_color_value)
        except ConfigurationError as exc:
            raise RequestError(422, "configuration", "背景颜色格式无效。") from exc
    else:
        raise RequestError(422, "configuration", "背景颜色必须使用十六进制格式。")

    try:
        return PixelizeConfig(
            width=_integer(raw, "width"),
            height=_integer(raw, "height"),
            max_colors=_integer(raw, "max_colors", 16),
            alpha_mode=_choice(raw, "alpha_mode", ("binary", "preserve"), "binary"),
            alpha_threshold=_number(raw, "alpha_threshold", 0.5),
            fit=_choice(raw, "fit", ("contain", "cover", "stretch"), "contain"),
            anchor=_choice(raw, "anchor", ("center", "bottom-center"), "center"),
            trim=_choice(raw, "trim", ("alpha", "none"), "alpha"),
            padding=_integer(raw, "padding", 0),
            dither=_choice(raw, "dither", ("none", "bayer4"), "none"),
            preview_scale=_integer(raw, "preview_scale", 8),
            palette=_colors(raw, "palette"),
            locked_colors=_colors(raw, "locked_colors"),
            key_color=key_color,
            key_tolerance=_number(raw, "key_tolerance", 0.0),
            max_input_pixels=MAX_INPUT_PIXELS,
            max_input_bytes=MAX_FILE_BYTES,
        )
    except RequestError:
        raise
    except ConfigurationError as exc:
        raise RequestError(
            422,
            "configuration",
            "参数组合不符合处理要求，请检查尺寸、颜色数量、留白和预览倍数。",
        ) from exc


def _zip_artifacts(artifacts: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(artifacts):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, artifacts[name])
    return output.getvalue()


def process_image(
    image_bytes: bytes,
    raw_config: Mapping[str, Any],
    mask_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    if not image_bytes:
        raise RequestError(400, "input", "请选择需要处理的源图片。")
    if len(image_bytes) > MAX_FILE_BYTES:
        raise RequestError(413, "input", "源图片不能超过 20 MB。")
    if mask_bytes is not None and len(mask_bytes) > MAX_FILE_BYTES:
        raise RequestError(413, "input", "蒙版图片不能超过 20 MB。")

    config = build_config(raw_config)
    try:
        result = pixelize_bytes(image_bytes, config, mask_bytes=mask_bytes)
    except ConfigurationError as exc:
        raise RequestError(422, "configuration", "参数组合不符合处理要求。") from exc
    except InputImageError as exc:
        raise RequestError(
            422,
            "input",
            "源图片无法处理，请确认它是有效的静态图片并包含可见内容。",
        ) from exc
    except QualityError as exc:
        raise RequestError(422, "quality", "生成结果未通过质量校验，请调整参数后重试。") from exc
    except ProcessingError as exc:
        raise RequestError(500, "processing", "图片处理未完成，请缩小图片后重试。") from exc
    except PixelizerError as exc:
        raise RequestError(500, "processing", "图片处理未完成。") from exc

    artifacts = result.artifact_bytes()
    return {
        "summary": {
            "size": "{}x{}".format(config.width, config.height),
            "width": config.width,
            "height": config.height,
            "maximum_colors": config.max_colors,
            "actual_color_count": result.palette_data["actual_color_count"],
            "alpha_mode": config.alpha_mode,
            "quality_passed": bool(result.report["passed"]),
            "preview_scale": config.preview_scale,
        },
        "warnings": list(result.warnings),
        "palette_data": result.palette_data,
        "report": result.report,
        "manifest": result.manifest,
        "artifacts": artifacts,
        "bundle": _zip_artifacts(artifacts),
        "preview_filename": result.preview_filename,
    }


def parse_multipart(content_type: str, body: bytes) -> Dict[str, UploadedPart]:
    try:
        header = content_type.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RequestError(415, "request", "请求格式无效。") from exc
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + header + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if message.get_content_type() != "multipart/form-data" or not message.is_multipart():
        raise RequestError(415, "request", "请使用 multipart/form-data 上传图片。")

    parts: Dict[str, UploadedPart] = {}
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        if name in parts:
            raise RequestError(400, "request", "上传字段 {} 不能重复。".format(name))
        payload = part.get_payload(decode=True)
        parts[name] = UploadedPart(
            name=name,
            data=payload if payload is not None else b"",
            filename=part.get_filename(),
            content_type=part.get_content_type(),
        )
    return parts


def _multipart_body(result: Mapping[str, Any]) -> Tuple[str, bytes]:
    boundary = "true-pixelizer-{}".format(secrets.token_hex(12))
    summary = {
        "ok": True,
        "summary": result["summary"],
        "warnings": result["warnings"],
        "palette_data": result["palette_data"],
        "report": result["report"],
        "manifest": result["manifest"],
    }
    artifacts = result["artifacts"]
    preview_filename = result["preview_filename"]
    response_parts = (
        ("result", None, "application/json", json.dumps(summary, ensure_ascii=False).encode("utf-8")),
        ("sprite", "sprite.png", "image/png", artifacts["sprite.png"]),
        ("preview", preview_filename, "image/png", artifacts[preview_filename]),
        ("palette", "palette.png", "image/png", artifacts["palette.png"]),
        ("report", "report.json", "application/json", artifacts["report.json"]),
        ("manifest", "manifest.json", "application/json", artifacts["manifest.json"]),
        ("bundle", "true-pixelizer-result.zip", "application/zip", result["bundle"]),
    )

    chunks = []
    for name, filename, content_type, payload in response_parts:
        chunks.append("--{}\r\n".format(boundary).encode("ascii"))
        disposition = 'Content-Disposition: form-data; name="{}"'.format(name)
        if filename:
            disposition += '; filename="{}"'.format(filename)
        chunks.append((disposition + "\r\n").encode("ascii"))
        chunks.append("Content-Type: {}\r\n\r\n".format(content_type).encode("ascii"))
        chunks.append(payload)
        chunks.append(b"\r\n")
    chunks.append("--{}--\r\n".format(boundary).encode("ascii"))
    return boundary, b"".join(chunks)


class PixelizerHandler(BaseHTTPRequestHandler):
    server_version = "TruePixelizer/{}".format(PACKAGE_VERSION)
    static_root = PROJECT_ROOT

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        payload: bytes,
        *,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control)
        self._security_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", encoded)

    def _send_error_payload(self, error: RequestError) -> None:
        self._send_json(
            error.status,
            {
                "ok": False,
                "error": {"type": error.error_type, "message": error.message},
            },
        )

    def _path(self) -> str:
        return urlsplit(self.path).path

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = self._path()
        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "service": "true-pixelizer", "version": PACKAGE_VERSION},
            )
            return
        if path not in ("/", "/index.html"):
            self._send_error_payload(RequestError(404, "not_found", "页面不存在。"))
            return
        try:
            payload = (self.static_root / "index.html").read_bytes()
        except OSError:
            self._send_error_payload(RequestError(500, "internal", "页面文件无法读取。"))
            return
        self._send_bytes(
            HTTPStatus.OK,
            "text/html; charset=utf-8",
            payload,
            cache_control="no-cache",
        )

    def do_POST(self) -> None:
        if self._path() != "/api/pixelize":
            self._send_error_payload(RequestError(404, "not_found", "接口不存在。"))
            return
        try:
            self._validate_origin()
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("multipart/form-data;"):
                raise RequestError(415, "request", "请使用表单上传图片。")
            content_length = self._content_length()
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                raise RequestError(400, "request", "上传内容不完整。")
            parts = parse_multipart(content_type, body)
            result = self._process_parts(parts)
            boundary, payload = _multipart_body(result)
            self._send_bytes(
                HTTPStatus.OK,
                "multipart/form-data; boundary={}".format(boundary),
                payload,
            )
        except RequestError as exc:
            self._send_error_payload(exc)
        except Exception:
            traceback.print_exc()
            self._send_error_payload(RequestError(500, "internal", "服务发生内部错误。"))

    def _validate_origin(self) -> None:
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if origin and (not host or origin.rstrip("/") != "http://{}".format(host)):
            raise RequestError(403, "origin", "只允许当前工具页面发起处理请求。")

    def _content_length(self) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise RequestError(411, "request", "请求缺少内容长度。")
        try:
            value = int(raw)
        except ValueError as exc:
            raise RequestError(400, "request", "内容长度无效。") from exc
        if value < 1:
            raise RequestError(400, "request", "上传内容为空。")
        if value > MAX_REQUEST_BYTES:
            raise RequestError(413, "request", "单次上传总大小不能超过 42 MB。")
        return value

    def _process_parts(self, parts: Mapping[str, UploadedPart]) -> Dict[str, Any]:
        image = parts.get("image")
        config_part = parts.get("config")
        if image is None or config_part is None:
            raise RequestError(400, "request", "必须同时提供源图片和处理参数。")
        if len(image.data) > MAX_FILE_BYTES:
            raise RequestError(413, "input", "源图片不能超过 20 MB。")
        mask = parts.get("mask")
        if mask is not None and len(mask.data) > MAX_FILE_BYTES:
            raise RequestError(413, "input", "蒙版图片不能超过 20 MB。")
        try:
            raw_config = json.loads(config_part.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError(400, "request", "处理参数不是有效 JSON。") from exc

        if not PROCESSING_SLOTS.acquire(blocking=False):
            raise RequestError(429, "busy", "当前已有两个任务在处理，请稍后重试。")
        try:
            return process_image(
                image.data,
                raw_config,
                mask_bytes=mask.data if mask is not None else None,
            )
        finally:
            PROCESSING_SLOTS.release()

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format_string % args))


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    static_root: Path = PROJECT_ROOT,
) -> ThreadingHTTPServer:
    class ConfiguredHandler(PixelizerHandler):
        pass

    ConfiguredHandler.static_root = Path(static_root)
    server = ThreadingHTTPServer((host, port), ConfiguredHandler)
    server.daemon_threads = True
    return server


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the True Pixelizer web tool.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8765)
    args = parser.parse_args(argv)
    server = create_server(args.host, args.port)
    address, port = server.server_address[:2]
    print("True Pixelizer 已启动：http://{}:{}/".format(address, port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
