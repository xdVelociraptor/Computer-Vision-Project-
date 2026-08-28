"""Shared infrastructure: logging, timing, IO, hashing, subprocess, determinism.

Everything in this module is pipeline-agnostic. Stage modules (preprocess,
colmap_pipeline, ...) import from here so that logging format, run manifests and
subprocess error reporting are identical across the whole project.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "CommandError",
    "CommandResult",
    "NumpyJSONEncoder",
    "Stopwatch",
    "StageTimer",
    "ensure_dir",
    "file_digest",
    "human_size",
    "iter_images",
    "prepare_output_dir",
    "read_json",
    "run_command",
    "set_global_seed",
    "setup_logging",
    "system_report",
    "timestamp",
    "write_csv",
    "write_json",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".webm"}

LOGGER_NAME = "sfm"


# ===========================================================================
# Filesystem helpers
# ===========================================================================


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Local-time stamp used for run directories and log file names."""
    return datetime.now().strftime(fmt)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_size(num_bytes: float) -> str:
    """Format a byte count for logs, e.g. ``1.4 GB``."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < step:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.1f} PB"


def iter_images(directory: Path, extensions: Optional[Iterable[str]] = None) -> List[Path]:
    """Return image files directly inside ``directory``, sorted by name.

    Sorting is what makes the pipeline deterministic: image order drives
    sequential matching, subset sampling and every reported index.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    exts = {e.lower() for e in (extensions or IMAGE_EXTENSIONS)}
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name.lower())


def prepare_output_dir(path: Path, force: bool, logger: Optional[logging.Logger] = None) -> Path:
    """Return an empty, ready-to-write ``path``.

    Refuses to clobber an existing non-empty directory unless ``force`` is set -
    this is the project's main guard against silently destroying results.
    """
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory '{path}' already exists and is not empty. "
                f"Re-run with --force to overwrite it, or choose another output path."
            )
        if logger:
            logger.warning("--force given: deleting existing contents of %s", path)
        shutil.rmtree(path)
    return ensure_dir(path)


def file_digest(path: Path, algorithm: str = "sha1", chunk_size: int = 1 << 20) -> str:
    """Streaming hash of a file (never loads the whole file into memory)."""
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def params_digest(payload: Mapping[str, Any]) -> str:
    """Stable short hash of a parameter dict, used as a cache key."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


# ===========================================================================
# Logging
# ===========================================================================


class _ColourFormatter(logging.Formatter):
    """Minimal ANSI colouring for console output (disabled when not a TTY)."""

    COLOURS = {
        "DEBUG": "\033[38;5;245m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;196m",
        "CRITICAL": "\033[48;5;196;38;5;231m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, use_colour: bool) -> None:
        super().__init__(fmt, datefmt="%H:%M:%S")
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.use_colour:
            colour = self.COLOURS.get(record.levelname)
            if colour:
                return f"{colour}{text}{self.RESET}"
        return text


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    name: str = LOGGER_NAME,
) -> logging.Logger:
    """Configure and return the project logger.

    Console output is concise; the optional file handler records DEBUG-level
    detail with full timestamps so that every run is auditable after the fact.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    use_colour = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(_ColourFormatter("[%(asctime)s] %(levelname)-8s %(message)s", use_colour))
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"
            )
        )
        logger.addHandler(file_handler)
        logger.debug("Logging to %s", log_file)

    return logger


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the project logger (already configured by ``setup_logging``)."""
    return logging.getLogger(name)


# ===========================================================================
# Timing
# ===========================================================================


class Stopwatch:
    """Context manager measuring wall-clock seconds.

    >>> with Stopwatch() as sw:
    ...     pass
    >>> sw.elapsed >= 0
    True
    """

    def __init__(self, label: str = "", logger: Optional[logging.Logger] = None) -> None:
        self.label = label
        self.logger = logger
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed = time.perf_counter() - self.start
        if self.logger and self.label:
            self.logger.info("%s finished in %.2f s", self.label, self.elapsed)


@dataclass
class StageTimer:
    """Accumulates named stage durations for the results table.

    The keys become ``feature_time``, ``matching_time``, ``mapping_time`` ...
    columns in ``results.csv``.
    """

    durations: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
        start = time.perf_counter()
        if logger:
            logger.info("--> stage '%s' started", name)
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            self.durations[name] = self.durations.get(name, 0.0) + dt
            if logger:
                logger.info("<-- stage '%s' took %.2f s", name, dt)

    @property
    def total(self) -> float:
        return float(sum(self.durations.values()))

    def as_dict(self, suffix: str = "_time") -> Dict[str, float]:
        out = {f"{k}{suffix}": round(v, 3) for k, v in self.durations.items()}
        out[f"total{suffix}"] = round(self.total, 3)
        return out


# ===========================================================================
# Serialisation
# ===========================================================================


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars/arrays, Paths and dataclasses."""

    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            value = float(o)
            return value if np.isfinite(value) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return o.as_posix()
        if isinstance(o, datetime):
            return o.isoformat()
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def write_json(path: Path, payload: Any, indent: int = 2) -> Path:
    """Atomically write ``payload`` as JSON (temp file + replace)."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, cls=NumpyJSONEncoder), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> Path:
    """Write dict rows to CSV with ``newline=""`` (avoids blank lines on Windows)."""
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    names = list(fieldnames) if fieldnames else list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return path


# ===========================================================================
# Determinism
# ===========================================================================


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs, and export ``PYTHONHASHSEED``.

    COLMAP determinism is handled separately, in :mod:`src.colmap_pipeline`,
    which passes ``--default_random_seed`` and ``--Mapper.random_seed``.

    Measured behaviour of COLMAP 4.1.1: seeding alone is **not** sufficient.
    Repeated multi-threaded runs of an identical dataset varied by roughly 0.2%
    in 3D point count, because bundle adjustment and triangulation depend on
    thread completion order. Setting ``runtime.num_threads=1`` in addition to
    the seeds produced bit-identical results across runs. See README section 11.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))


# ===========================================================================
# Subprocess execution (COLMAP CLI and friends)
# ===========================================================================


@dataclass
class CommandResult:
    """Outcome of an external command, including full captured streams."""

    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail(self, n: int = 40) -> str:
        """Last ``n`` lines of stderr (or stdout when stderr is empty)."""
        stream = self.stderr.strip() or self.stdout.strip()
        return "\n".join(stream.splitlines()[-n:])


class CommandError(RuntimeError):
    """Raised when an external command exits non-zero.

    Carries the :class:`CommandResult` so callers can log stdout/stderr and
    attach a diagnosis to the failure report.
    """

    def __init__(self, result: CommandResult, hint: str = "") -> None:
        self.result = result
        self.hint = hint
        cmd = " ".join(result.command)
        message = f"Command failed with exit code {result.returncode}:\n  {cmd}"
        if hint:
            message += f"\nProbable cause: {hint}"
        tail = result.tail()
        if tail:
            message += f"\n--- last output ---\n{tail}"
        super().__init__(message)


def run_command(
    command: Sequence[str],
    logger: Optional[logging.Logger] = None,
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
    timeout: Optional[float] = None,
    log_path: Optional[Path] = None,
    hint_fn: Optional[Any] = None,
) -> CommandResult:
    """Run an external command, capturing and logging both streams.

    Parameters
    ----------
    command:
        Argument vector. Never a shell string - this avoids Windows quoting bugs
        with paths that contain spaces (``OneDrive\\Desktop\\...``).
    check:
        Raise :class:`CommandError` on a non-zero exit code.
    log_path:
        If given, the full stdout/stderr of this invocation is written there.
    hint_fn:
        ``callable(CommandResult) -> str`` producing a probable-cause diagnosis;
        used by the COLMAP wrapper to explain failures.
    """
    cmd = [str(c) for c in command]
    log = logger or get_logger()
    log.debug("Executing: %s", " ".join(cmd))

    start = time.perf_counter()
    try:
        proc = subprocess.run(  # noqa: S603 - argument vector, no shell
            cmd,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        duration = time.perf_counter() - start
        result = CommandResult(cmd, 127, "", str(exc), duration)
        raise CommandError(
            result,
            f"Executable '{cmd[0]}' was not found on PATH. "
            f"Install it, or set its absolute path in the configuration.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        result = CommandResult(cmd, 124, exc.stdout or "", exc.stderr or "", duration)
        raise CommandError(result, f"Command exceeded the {timeout}s timeout.") from exc

    duration = time.perf_counter() - start
    result = CommandResult(cmd, proc.returncode, proc.stdout or "", proc.stderr or "", duration)

    if log_path is not None:
        log_path = Path(log_path)
        ensure_dir(log_path.parent)
        log_path.write_text(
            f"$ {' '.join(cmd)}\n"
            f"# exit={result.returncode} duration={duration:.2f}s at {utc_now_iso()}\n"
            f"\n===== STDOUT =====\n{result.stdout}\n"
            f"\n===== STDERR =====\n{result.stderr}\n",
            encoding="utf-8",
        )

    if result.ok:
        log.debug("Command succeeded in %.2f s", duration)
    else:
        log.error("Command failed (exit %d) after %.2f s: %s", result.returncode, duration, " ".join(cmd))
        if result.stderr.strip():
            log.error("stderr tail:\n%s", result.tail())
        if check:
            hint = hint_fn(result) if hint_fn else ""
            raise CommandError(result, hint)

    return result


# ===========================================================================
# Environment / provenance
# ===========================================================================


def system_report() -> Dict[str, Any]:
    """Hardware and OS facts recorded alongside every run for reproducibility."""
    report: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil  # imported lazily: optional dependency

        report["cpu_count_physical"] = psutil.cpu_count(logical=False)
        report["ram_total"] = human_size(psutil.virtual_memory().total)
    except Exception:  # pragma: no cover - psutil is optional
        report["cpu_count_physical"] = None
        report["ram_total"] = None
    return report


def disk_free(path: Path) -> Dict[str, Any]:
    """Free/total space for the volume holding ``path``."""
    path = Path(path)
    probe = path if path.exists() else path.anchor or Path.cwd()
    usage = shutil.disk_usage(probe)
    return {
        "path": str(probe),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "total_human": human_size(usage.total),
        "free_human": human_size(usage.free),
    }
