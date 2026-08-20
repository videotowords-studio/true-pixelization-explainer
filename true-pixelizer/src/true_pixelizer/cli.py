import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .colors import RGB, parse_hex_color
from .config import PixelizeConfig
from .errors import (
    ConfigurationError,
    ExportError,
    InputImageError,
    PixelizerError,
    ProcessingError,
    QualityError,
)
from .exporter import write_artifacts
from .pipeline import pixelize_bytes
from .quality import decode_png_rgba, validate_rgba
from .version import PACKAGE_VERSION


class _ArgumentParsingError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentParsingError(message)


def _parse_size(value: str) -> Tuple[int, int]:
    normalized = value.strip().lower()
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must use WIDTHxHEIGHT, for example 32x32.")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Size must use positive integer dimensions."
        ) from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("Size dimensions must be positive.")
    return width, height


def _read_bytes(path: Path, label: str, max_bytes: int) -> bytes:
    try:
        if not path.is_file():
            raise InputImageError("The {} file does not exist: {}".format(label, path))
        if path.stat().st_size > max_bytes:
            raise InputImageError(
                "The {} file exceeds the configured byte safety limit.".format(label)
            )
        payload = path.read_bytes()
        if len(payload) > max_bytes:
            raise InputImageError(
                "The {} file exceeds the configured byte safety limit.".format(label)
            )
        return payload
    except InputImageError:
        raise
    except OSError as exc:
        raise InputImageError(
            "The {} file could not be read: {}".format(label, path)
        ) from exc


def _parse_color_list(values: Sequence[str]) -> Tuple[RGB, ...]:
    colors: List[RGB] = []
    for value in values:
        for item in value.split(","):
            if item.strip():
                colors.append(parse_hex_color(item))
    return tuple(colors)


def _read_palette_file(path: Path) -> Tuple[RGB, ...]:
    try:
        if path.stat().st_size > 1_000_000:
            raise ConfigurationError(
                "The palette file may contain at most 1,000,000 bytes."
            )
        payload = path.read_bytes()
        if len(payload) > 1_000_000:
            raise ConfigurationError(
                "The palette file may contain at most 1,000,000 bytes."
            )
        raw = json.loads(payload.decode("utf-8"))
    except ConfigurationError:
        raise
    except OSError as exc:
        raise ConfigurationError(
            "The palette file could not be read: {}".format(path)
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("The palette file must contain valid JSON.") from exc
    if isinstance(raw, dict):
        raw = raw.get("colors")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ConfigurationError(
            'The palette JSON must be an array of colors or {"colors": [...]}.'
        )
    return tuple(parse_hex_color(item) for item in raw)


def _resolve_palette(args: argparse.Namespace) -> Tuple[RGB, ...]:
    inline = _parse_color_list(args.palette or [])
    if args.palette_file is not None:
        if inline:
            raise ConfigurationError(
                "Use either --palette or --palette-file, not both."
            )
        return _read_palette_file(args.palette_file)
    return inline


def _config_from_args(
    args: argparse.Namespace, *, include_processing: bool
) -> PixelizeConfig:
    width, height = args.size
    palette = _resolve_palette(args)
    locked = _parse_color_list(args.locked_color or [])
    if include_processing:
        return PixelizeConfig(
            width=width,
            height=height,
            max_colors=args.max_colors,
            alpha_mode=args.alpha,
            alpha_threshold=args.alpha_threshold,
            fit=args.fit,
            anchor=args.anchor,
            trim=args.trim,
            padding=args.padding,
            dither=args.dither,
            preview_scale=args.preview_scale,
            palette=palette,
            locked_colors=locked,
            key_color=parse_hex_color(args.key_color) if args.key_color else None,
            key_tolerance=args.key_tolerance,
            max_input_pixels=args.max_input_pixels,
            max_input_bytes=args.max_input_bytes,
        )
    return PixelizeConfig(
        width=width,
        height=height,
        max_colors=args.max_colors,
        alpha_mode=args.alpha,
        alpha_threshold=args.alpha_threshold,
        preview_scale=1,
        palette=palette,
        locked_colors=locked,
    )


def _add_palette_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--palette",
        action="append",
        help='Fixed colors as a comma-separated list, for example "#000000,#FFFFFF".',
    )
    parser.add_argument(
        "--palette-file",
        type=Path,
        help="JSON file containing an array of fixed hexadecimal colors.",
    )
    parser.add_argument(
        "--locked-color",
        action="append",
        help="Color that the automatic palette must retain; may be repeated.",
    )


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", required=True, type=_parse_size)
    parser.add_argument("--max-colors", type=int, default=16)
    parser.add_argument("--alpha", choices=("binary", "preserve"), default="binary")
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    _add_palette_arguments(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="pixelize",
        description="Convert a static image into a validated logical pixel grid.",
    )
    parser.add_argument("--version", action="version", version=PACKAGE_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pixelize_parser = subparsers.add_parser(
        "pixelize", help="Create a sprite, preview, palette, and quality report."
    )
    pixelize_parser.add_argument("input", type=Path)
    pixelize_parser.add_argument("--output-dir", required=True, type=Path)
    _add_contract_arguments(pixelize_parser)
    pixelize_parser.add_argument(
        "--fit", choices=("contain", "cover", "stretch"), default="contain"
    )
    pixelize_parser.add_argument(
        "--anchor", choices=("center", "bottom-center"), default="center"
    )
    pixelize_parser.add_argument("--trim", choices=("alpha", "none"), default="alpha")
    pixelize_parser.add_argument("--padding", type=int, default=0)
    pixelize_parser.add_argument("--dither", choices=("none", "bayer4"), default="none")
    pixelize_parser.add_argument("--preview-scale", type=int, default=8)
    pixelize_parser.add_argument("--mask", type=Path)
    pixelize_parser.add_argument("--key-color")
    pixelize_parser.add_argument("--key-tolerance", type=float, default=0.0)
    pixelize_parser.add_argument("--max-input-pixels", type=int, default=16_000_000)
    pixelize_parser.add_argument("--max-input-bytes", type=int, default=100_000_000)
    pixelize_parser.add_argument("--force", action="store_true")
    pixelize_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate", help="Check an existing sprite against a pixel contract."
    )
    validate_parser.add_argument("input", type=Path)
    _add_contract_arguments(validate_parser)
    validate_parser.add_argument("--json", action="store_true")
    return parser


def _print_payload(payload: Dict[str, Any], as_json: bool, error: bool = False) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    stream = sys.stderr if error else sys.stdout
    if payload.get("ok"):
        if payload.get("command") == "pixelize":
            print("像素化完成：{}".format(payload["output_dir"]), file=stream)
            print(
                "尺寸 {}，实际使用 {} 色，质量校验通过。".format(
                    payload["summary"]["size"],
                    payload["summary"]["actual_color_count"],
                ),
                file=stream,
            )
        else:
            print("质量校验通过。", file=stream)
    else:
        message = payload.get("error", {}).get("message", "Operation failed.")
        print(message, file=stream)


def _ensure_inputs_are_not_output_targets(
    args: argparse.Namespace, config: PixelizeConfig
) -> None:
    output_dir = args.output_dir.resolve(strict=False)
    artifact_names = {
        "sprite.png",
        "preview-{}x.png".format(config.preview_scale),
        "palette.png",
        "palette.json",
        "report.json",
        "manifest.json",
    }
    target_paths = [output_dir / name for name in artifact_names]
    targets = {path.resolve(strict=False) for path in target_paths}
    protected = [args.input]
    if args.mask is not None:
        protected.append(args.mask)
    if args.palette_file is not None:
        protected.append(args.palette_file)
    for path in protected:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise InputImageError(
                "An input dependency could not be resolved: {}".format(path)
            ) from exc
        same_existing_file = False
        for target in target_paths:
            if target.exists() or target.is_symlink():
                try:
                    if os.path.samefile(str(resolved), str(target)):
                        same_existing_file = True
                        break
                except OSError as exc:
                    raise InputImageError(
                        "An output target could not be resolved safely: {}".format(
                            target
                        )
                    ) from exc
        if same_existing_file or resolved in targets:
            raise ConfigurationError(
                "An input dependency must not share a path with an output artifact: {}".format(
                    resolved
                )
            )


def _run_pixelize(args: argparse.Namespace) -> int:
    config = _config_from_args(args, include_processing=True)
    _ensure_inputs_are_not_output_targets(args, config)
    image_bytes = _read_bytes(args.input, "input image", config.max_input_bytes)
    mask_bytes = (
        _read_bytes(args.mask, "mask", config.max_input_bytes) if args.mask else None
    )
    result = pixelize_bytes(image_bytes, config, mask_bytes=mask_bytes)
    artifacts = result.artifact_bytes()
    descriptions = write_artifacts(args.output_dir, artifacts, force=args.force)
    output_dir = args.output_dir.resolve()
    described_with_paths = {
        name: dict(description, path=str(output_dir / name))
        for name, description in descriptions.items()
    }
    payload = {
        "ok": True,
        "command": "pixelize",
        "output_dir": str(output_dir),
        "artifacts": described_with_paths,
        "summary": {
            "size": "{}x{}".format(config.width, config.height),
            "maximum_colors": config.max_colors,
            "actual_color_count": result.palette_data["actual_color_count"],
            "alpha_mode": config.alpha_mode,
            "quality_passed": result.report["passed"],
        },
        "warnings": list(result.warnings),
    }
    _print_payload(payload, args.json)
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    config = _config_from_args(args, include_processing=False)
    image_bytes = _read_bytes(args.input, "sprite", config.max_input_bytes)
    rgba = decode_png_rgba(image_bytes, config.max_input_pixels)
    report = validate_rgba(
        rgba,
        config,
        allowed_palette=config.palette if config.palette else (),
    )
    payload = {
        "ok": bool(report["passed"]),
        "command": "validate",
        "input": str(args.input.resolve()),
        "report": report,
    }
    if not report["passed"]:
        payload["error"] = {
            "type": "quality",
            "message": "The sprite does not satisfy the requested pixel contract.",
            "exit_code": 4,
        }
    _print_payload(payload, args.json, error=not report["passed"])
    return 0 if report["passed"] else 4


def _error_payload(
    error: Exception, exit_code: int, error_type: str
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": error_type,
            "message": str(error),
            "exit_code": exit_code,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    as_json = "--json" in arguments
    parser = _build_parser()
    try:
        args = parser.parse_args(arguments)
    except _ArgumentParsingError as exc:
        payload = _error_payload(exc, 2, "arguments")
        if not as_json:
            parser.print_usage(sys.stderr)
        _print_payload(payload, as_json, error=True)
        return 2
    as_json = bool(getattr(args, "json", False))
    try:
        if args.command == "pixelize":
            return _run_pixelize(args)
        return _run_validate(args)
    except (ConfigurationError, InputImageError) as exc:
        error_type = "configuration" if isinstance(exc, ConfigurationError) else "input"
        _print_payload(_error_payload(exc, 2, error_type), as_json, error=True)
        return 2
    except QualityError as exc:
        _print_payload(_error_payload(exc, 4, "quality"), as_json, error=True)
        return 4
    except ProcessingError as exc:
        _print_payload(_error_payload(exc, 3, "processing"), as_json, error=True)
        return 3
    except ExportError as exc:
        _print_payload(_error_payload(exc, 3, "export"), as_json, error=True)
        return 3
    except PixelizerError as exc:
        _print_payload(_error_payload(exc, 3, "processing"), as_json, error=True)
        return 3
    except Exception:  # pragma: no cover - defensive CLI boundary
        internal = RuntimeError("Internal processing error.")
        _print_payload(_error_payload(internal, 3, "internal"), as_json, error=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
