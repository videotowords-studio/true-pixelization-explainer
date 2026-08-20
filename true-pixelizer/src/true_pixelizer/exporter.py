import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .errors import ConfigurationError, ExportError


ArtifactDescriptions = Dict[str, Dict[str, Any]]


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def describe_artifacts(artifacts: Mapping[str, bytes]) -> ArtifactDescriptions:
    validated = _validate_artifacts(artifacts)
    return {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for name, payload in validated
    }


def write_artifacts(
    output_dir: Path,
    artifacts: Mapping[str, bytes],
    force: bool = False,
) -> ArtifactDescriptions:
    output_dir = Path(output_dir)
    if output_dir.name in ("", ".", ".."):
        raise ConfigurationError("output_dir must name a dedicated directory.")

    validated = _validate_artifacts(artifacts)
    descriptions = describe_artifacts(dict(validated))
    parent = output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        _validate_output_dir(output_dir, force)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(parent))
        )

        try:
            for name, payload in validated:
                staged_path = temporary_dir / name
                with staged_path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            _fsync_directory(temporary_dir)

            _validate_output_dir(output_dir, force)
            if not output_dir.exists() or not any(output_dir.iterdir()):
                os.replace(str(temporary_dir), str(output_dir))
                _fsync_directory(parent)
                temporary_dir = None
            else:
                _replace_with_rollback(output_dir, temporary_dir, validated)
                _fsync_directory(parent)

            return descriptions
        finally:
            if temporary_dir is not None:
                shutil.rmtree(temporary_dir, ignore_errors=True)
    except ConfigurationError:
        raise
    except ExportError:
        raise
    except OSError as exc:
        raise ExportError("The output artifacts could not be written safely.") from exc


def _replace_with_rollback(
    output_dir: Path,
    temporary_dir: Path,
    validated: Tuple[Tuple[str, bytes], ...],
) -> None:
    names = [name for name, _payload in validated]
    names.sort(key=lambda name: (name == "manifest.json", name))
    for name in names:
        target = output_dir / name
        if target.is_dir() and not target.is_symlink():
            raise ConfigurationError(f"Cannot replace directory with artifact: {name}")

    backup_dir = temporary_dir / ".backup"
    discarded_dir = temporary_dir / ".discarded"
    backup_dir.mkdir()
    discarded_dir.mkdir()
    touched = []
    try:
        for name in names:
            if name == "manifest.json":
                _fsync_directory(output_dir)
            target = output_dir / name
            backup = backup_dir / name
            if target.exists() or target.is_symlink():
                os.replace(str(target), str(backup))
            touched.append(name)
            os.replace(str(temporary_dir / name), str(target))
        _fsync_directory(output_dir)
    except BaseException:
        rollback_error = None
        for name in reversed(touched):
            target = output_dir / name
            backup = backup_dir / name
            try:
                if target.exists() or target.is_symlink():
                    os.replace(str(target), str(discarded_dir / name))
                if backup.exists() or backup.is_symlink():
                    os.replace(str(backup), str(target))
            except OSError as exc:
                rollback_error = exc
        try:
            _fsync_directory(output_dir)
        except OSError as exc:
            rollback_error = rollback_error or exc
        if rollback_error is not None:
            raise ExportError(
                "The output update failed and could not be fully rolled back."
            ) from rollback_error
        raise


def _validate_artifacts(
    artifacts: Mapping[str, bytes],
) -> Tuple[Tuple[str, bytes], ...]:
    validated = []
    for name, payload in artifacts.items():
        if not isinstance(name, str) or not _is_simple_filename(name):
            raise ConfigurationError(
                "Artifact names must be simple filenames without path separators."
            )
        if not isinstance(payload, bytes):
            raise ConfigurationError(f"Artifact payload must be bytes: {name}")
        validated.append((name, payload))
    return tuple(sorted(validated, key=lambda item: item[0]))


def _is_simple_filename(name: str) -> bool:
    return (
        bool(name)
        and name not in (".", "..")
        and "\x00" not in name
        and "/" not in name
        and "\\" not in name
        and Path(name).name == name
    )


def _validate_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.is_symlink():
        raise ConfigurationError("output_dir must not be a symbolic link.")
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigurationError("output_dir must be a directory.")
    if output_dir.exists() and not force and any(output_dir.iterdir()):
        raise ConfigurationError(
            "output_dir is not empty; use force to replace matching artifacts."
        )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
