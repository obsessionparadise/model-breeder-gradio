"""
Model Breeder — Resource guards
=================================
Everything in this module exists to make one promise true: this app should
never OOM-crash, silently corrupt output, or let two heavy jobs race for
the same RAM/VRAM. See CRITIQUE.md for the audit that produced each guard
here — every function below maps to one or more numbered findings there.

Deliberately dependency-light: RAM/disk introspection works from stdlib
(`/proc/meminfo`, `shutil.disk_usage`) with an optional `psutil` path for
platforms where `/proc` doesn't exist (Windows, some Colab variants), so
this module never becomes the reason a fresh environment fails to start.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import psutil  # optional; used as a fallback / cross-platform path
except ImportError:
    psutil = None

import torch


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

class InsufficientResourceError(RuntimeError):
    """Raised when a preflight check determines an operation would not fit
    in available RAM or disk space. Always raised *before* any destructive
    work starts."""


class GpuOutOfMemoryError(RuntimeError):
    """Raised when a CUDA allocation fails during model load or generation.
    Always raised after the CUDA cache has already been cleared, so the
    session is left in a recoverable state."""


class OperationCancelled(RuntimeError):
    """Raised when a CancelToken is tripped mid-operation. Callers treat
    this as a clean stop, not an error: any partial output file is deleted
    the same way it would be for a real failure."""


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# RAM introspection (CRITIQUE.md A1, A2)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

def available_ram_bytes() -> Optional[int]:
    """Best-effort available RAM in bytes. Returns None if it genuinely
    cannot be determined (never raises) — callers must treat None as
    'unknown, proceed with a warning' rather than 'zero'."""
    try:
        with open('/proc/meminfo') as f:
            info = {}
            for line in f:
                parts = line.split(':')
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                m = re.search(r'(\d+)', parts[1])
                if m:
                    info[key] = int(m.group(1)) * 1024  # kB -> bytes
        if 'MemAvailable' in info:
            return info['MemAvailable']
        if 'MemFree' in info and 'Cached' in info:
            return info['MemFree'] + info['Cached']
    except Exception:
        pass
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
    return None


def total_ram_bytes() -> Optional[int]:
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    m = re.search(r'(\d+)', line)
                    if m:
                        return int(m.group(1)) * 1024
    except Exception:
        pass
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().total)
        except Exception:
            pass
    return None


def require_ram(min_free_mb: float, context: str) -> None:
    """Raise InsufficientResourceError if fewer than min_free_mb are
    available. Silently passes (with no guarantee) if availability can't
    be determined at all — an unknown environment shouldn't hard-block
    every operation, only a *known* shortfall should."""
    avail = available_ram_bytes()
    if avail is None:
        return
    min_free = min_free_mb * 1024 * 1024
    if avail < min_free:
        raise InsufficientResourceError(
            f'Not enough free RAM for {context}: needs ~{min_free_mb:.0f} MB, '
            f'only {avail / 1e6:.0f} MB available. Close other applications/'
            f'notebook kernels, or use a machine with more RAM.')


def estimate_ckpt_ram_mb(path) -> float:
    """Conservative RAM estimate for loading a file as a full in-memory
    dict (torch.load / safetensors.load_file) rather than streaming it.
    Assumes worst case: file is read, then upcast to fp32 (2x for fp16
    source), plus ~15% Python/tensor-object overhead."""
    size_mb = Path(path).stat().st_size / 1e6
    return size_mb * 2.0 * 1.15


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Disk introspection (CRITIQUE.md B1)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

def check_disk_space(target_dir, required_bytes: float, context: str, margin: float = 1.15) -> None:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(target_dir)).free
    needed = required_bytes * margin
    if free < needed:
        raise InsufficientResourceError(
            f'Not enough free disk space for {context}: needs ~{needed / 1e9:.2f} GB '
            f'(with safety margin), only {free / 1e9:.2f} GB free on {target_dir}.')


def estimate_output_bytes(*input_paths, fp16: bool = True) -> int:
    """Rough output-size estimate for a merge/bake: sum of input sizes,
    scaled by ~0.55 if converting to fp16 (outputs are rarely larger than
    the biggest single input regardless of how many inputs there are, since
    merges are 1:1 tensor combinations, not concatenations)."""
    sizes = [Path(p).stat().st_size for p in input_paths if p and Path(p).exists()]
    if not sizes:
        return 0
    base = max(sizes)
    return int(base * (0.55 if fp16 else 1.05))


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Single-flight job lock (CRITIQUE.md C2)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

class JobLock:
    """A non-blocking, single-flight lock shared by every heavy operation
    in this app (merge / bake / vae-bake / blend / checkpoint-load /
    generate). Only one such job may run at a time, process-wide — this is
    the single biggest OOM-prevention measure here: two multi-GB streaming
    operations running concurrently is the most reliable way to exhaust
    RAM/VRAM even when each one individually fits comfortably."""

    def __init__(self):
        self._lock = threading.Lock()
        self._holder = None

    @contextlib.contextmanager
    def acquire(self, job_name: str):
        got = self._lock.acquire(blocking=False)
        if not got:
            raise InsufficientResourceError(
                f'Another job ("{self._holder}") is already running. Wait for it to '
                f'finish (or cancel it) before starting "{job_name}" — running two '
                f'heavy jobs at once is the most common cause of out-of-memory crashes.')
        self._holder = job_name
        try:
            yield
        finally:
            self._holder = None
            self._lock.release()

    @property
    def busy(self) -> bool:
        return self._lock.locked()


# One process-wide lock, imported by both engine.py and app.py.
JOB_LOCK = JobLock()


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Cancellation (CRITIQUE.md D3)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

class CancelToken:
    """A cooperative cancellation flag. Long-running loops call .check()
    at safe boundaries (between tensors, between denoising steps); it
    raises OperationCancelled if .cancel() was called from the UI thread."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def check(self):
        if self._event.is_set():
            raise OperationCancelled('Cancelled by user.')

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# GPU OOM handling (CRITIQUE.md A5, A6)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

def _is_cuda_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


@contextlib.contextmanager
def gpu_oom_guard(context: str):
    """Wrap any CUDA-touching call. On OOM: empty the cache, collect
    garbage, and raise a friendly GpuOutOfMemoryError so the session stays
    usable for a retry at lower settings instead of being left corrupted."""
    try:
        yield
    except Exception as e:
        if _is_cuda_oom(e):
            import gc as _gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _gc.collect()
            raise GpuOutOfMemoryError(
                f'Ran out of GPU memory during {context}. The GPU cache has been '
                f'cleared so you can retry — try a lower resolution, fewer images '
                f'per batch, or fewer steps. If this keeps happening, the checkpoint '
                f'may simply be too large for this GPU.') from e
        raise


def release_gpu_memory():
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Generation-request clamping (CRITIQUE.md A7)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

GEN_LIMITS = {
    'width': (64, 2048), 'height': (64, 2048),
    'steps': (1, 150), 'cfg': (0.0, 30.0),
    'n_images': (1, 8), 'clip_skip': (1, 12),
}


def clamp_generation_request(width, height, steps, cfg, n_images, clip_skip, log_cb=None):
    def _clamp(name, value, lo, hi):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = lo
        clamped = max(lo, min(hi, v))
        if clamped != v and log_cb:
            log_cb(f'⚠ {name}={value} out of allowed range [{lo}, {hi}] — clamped to {clamped}')
        return clamped

    w_lo, w_hi = GEN_LIMITS['width']
    h_lo, h_hi = GEN_LIMITS['height']
    s_lo, s_hi = GEN_LIMITS['steps']
    c_lo, c_hi = GEN_LIMITS['cfg']
    n_lo, n_hi = GEN_LIMITS['n_images']
    cs_lo, cs_hi = GEN_LIMITS['clip_skip']
    return (
        int(_clamp('width', width, w_lo, w_hi)),
        int(_clamp('height', height, h_lo, h_hi)),
        int(_clamp('steps', steps, s_lo, s_hi)),
        _clamp('cfg', cfg, c_lo, c_hi),
        int(_clamp('n_images', n_images, n_lo, n_hi)),
        int(_clamp('clip_skip', clip_skip, cs_lo, cs_hi)),
    )


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Input sanitization (CRITIQUE.md F1, F2)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

def safe_filename(name: str, default: str = 'output.safetensors') -> str:
    name = (name or '').strip()
    if not name:
        return default
    # Strip any path components — keep only the final segment — then drop
    # characters that aren't safe across Linux/macOS/Windows filesystems.
    name = Path(name).name
    name = re.sub(r'[^A-Za-z0-9._\-() ]', '_', name)
    name = name.lstrip('.').strip()
    return name or default


def sanity_check_block_weights(vals, log_cb=None, lo=-3.0, hi=3.0):
    if not vals:
        return
    out_of_range = [v for v in vals if v < lo or v > hi]
    if out_of_range and log_cb:
        log_cb(f'⚠ {len(out_of_range)} block weight(s) are outside the typical '
               f'[{lo}, {hi}] range ({out_of_range[:5]}{"..." if len(out_of_range) > 5 else ""}) '
               f'— this is allowed but double-check it wasn\'t a typo.')


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Approximate latent -> RGB preview (CRITIQUE.md D3)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Small linear projection matrices from latent channels straight to RGB,
# skipping a real VAE decode entirely — this is the same category of
# technique long used across the open-source SD ecosystem (e.g.
# AUTOMATIC1111's/ComfyUI's "approx VAE" live preview) for a cheap in-
# progress thumbnail. It is a blurry, approximate color sketch, not a
# faithful preview — that's the accepted trade for being nearly free to
# compute every few denoising steps.
_LATENT_RGB_FACTORS = {
    4: [  # SD1.5 / SDXL (4-channel UNet latents)
        [0.3512, 0.2297, 0.3227],
        [0.3250, 0.4974, 0.2350],
        [-0.2829, 0.1762, 0.2721],
        [-0.2120, -0.2616, -0.7177],
    ],
    16: [  # Flux / Z-Image / Krea2(Qwen-Image) / Anima — 16-channel latents
        [-0.0346, 0.0244, 0.0681], [0.0034, 0.0210, 0.0687],
        [0.0275, -0.0668, -0.0433], [-0.0174, 0.0160, 0.0617],
        [0.0859, 0.0721, 0.0329], [0.0004, 0.0383, 0.0115],
        [0.0405, 0.0861, 0.0915], [-0.0236, -0.0185, -0.0259],
        [-0.0245, 0.0250, 0.1180], [0.1008, 0.0755, -0.0421],
        [-0.0515, 0.0201, 0.0011], [0.0428, -0.0012, -0.0161],
        [0.0187, 0.0091, 0.1200], [0.0148, 0.0364, -0.0231],
        [-0.0417, 0.0141, -0.0272], [-0.0407, 0.0067, -0.0349],
    ],
}


def approx_latents_to_preview(latents):
    """latents: a torch tensor [B, C, H, W] (or [C, H, W]). Returns a PIL
    Image for the first item in the batch, or None if the channel count
    isn't one this app has a projection matrix for (never raises — a
    missing preview must never break generation itself)."""
    try:
        import numpy as np
        from PIL import Image
        t = latents
        if t.dim() == 3:
            t = t.unsqueeze(0)
        c = t.shape[1]
        factors = _LATENT_RGB_FACTORS.get(c)
        if factors is None:
            return None
        weight = torch.tensor(factors, dtype=torch.float32, device='cpu').T  # [3, C]
        x = t[0].detach().to('cpu', torch.float32)  # [C, H, W]
        rgb = torch.einsum('chw,rc->rhw', x, weight)  # [3, H, W]
        rgb = (rgb / 2 + 0.5).clamp(0, 1)
        arr = (rgb.permute(1, 2, 0).numpy() * 255).astype('uint8')
        return Image.fromarray(arr)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# Atomic file write helper (CRITIQUE.md C3)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

def atomic_copy_into(src_path, dest_dir) -> Path:
    """Copy src_path into dest_dir under a temp name, then atomically
    rename to the final name — so a concurrent reader of dest_dir never
    observes a partially-written file.

    Two things this specifically guards against, both found by testing
    rather than assumed:
      - The temp filename deliberately does NOT end in a recognized model
        extension (.safetensors/.ckpt/.pt/...). pathlib's glob('*'), unlike
        shell globbing, DOES match dotfiles — a temp name that preserved
        the original extension (e.g. '.upload_123_model.safetensors') would
        still match a '*.safetensors' glob and could appear as a selectable
        checkpoint mid-upload, especially for large files where the copy
        takes a while. Verified this concretely before fixing it: the old
        naming scheme did leak into list_checkpoints() during the copy
        window; the '.partial' suffix here can never match any of the
        extension-specific globs those listing functions use.
      - A disk-full (or any other) failure partway through the copy cleans
        up its own temp file rather than leaving debris behind silently.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_name = safe_filename(Path(src_path).name)
    final_path = dest_dir / final_name
    tmp_path = dest_dir / f'.upload_{os.getpid()}_{int(time.time() * 1000)}_{final_name}.partial'
    try:
        src_size = Path(src_path).stat().st_size
        check_disk_space(dest_dir, src_size, f'uploading {final_name}')
        shutil.copy(str(src_path), str(tmp_path))
        os.replace(str(tmp_path), str(final_path))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise
    return final_path


def resolve_hf_cache_dir() -> Path:
    """Best-effort resolution of where huggingface_hub will actually
    download model files to, respecting the same environment variables it
    does (checked in the same precedence order the library itself uses)."""
    if os.environ.get('HF_HOME'):
        return Path(os.environ['HF_HOME']) / 'hub'
    if os.environ.get('HUGGINGFACE_HUB_CACHE'):
        return Path(os.environ['HUGGINGFACE_HUB_CACHE'])
    return Path.home() / '.cache' / 'huggingface' / 'hub'


def check_hf_repo_disk_space(repo_id_or_path: str, log_cb=None) -> None:
    """Best-effort preflight for the newer architectures (Z-Image/Krea2/
    Anima), which download a multi-GB diffusers repo on first use. The RAM
    check elsewhere in this module doesn't cover this at all — a HF Hub
    download lands in the HF cache directory, which is frequently a
    *different filesystem* than wherever MODEL_BREEDER_DIR points, so a
    disk check against the workspace folder wouldn't catch it. Krea 2 alone
    is ~13B parameters (~25GB+ in bf16) before its text encoder and VAE.

    Deliberately never blocks on failing to *determine* the size (gated
    repos requiring authentication, network hiccups, huggingface_hub not
    importable) — only blocks when the size IS known and genuinely won't
    fit; refusing to even attempt the check would be worse than proceeding
    with an unverified download, since most of the time it'll work fine."""
    if Path(repo_id_or_path).exists():
        return  # a local folder, not a Hub download — nothing to preflight
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id_or_path, files_metadata=True)
        total = sum((getattr(s, 'size', 0) or 0) for s in (info.siblings or []))
        if total <= 0:
            if log_cb:
                log_cb(f'  (could not determine download size for {repo_id_or_path} — skipping disk preflight)')
            return
        cache_dir = resolve_hf_cache_dir()
        if log_cb:
            log_cb(f'  Estimated download size: {total / 1e9:.1f} GB -> {cache_dir}')
        check_disk_space(cache_dir, total, f'downloading {repo_id_or_path}', margin=1.05)
    except InsufficientResourceError:
        raise
    except Exception as e:
        if log_cb:
            log_cb(f'  (could not preflight-check download size for {repo_id_or_path}: {e} — proceeding anyway)')


# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

# System status snapshot (surfaced in the UI so the guards are visible,
# not just invisible plumbing)
# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: safe atomic copy with filename validation
# ---------------------------------------------------------------------------

def safe_copy_into(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Performs an atomic copy using a temporary file.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    dest_path = dest_dir / safe_name
    # Perform atomic copy: write to a temporary file then replace
    temp_path = dest_path.with_suffix('.tmp')
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dest_path)

# ---------------------------------------------------------------------------
# Typed exceptions — so UI code can distinguish "expected, show the message"
# from "unexpected, show the full traceback" (CRITIQUE.md B3).
# ---------------------------------------------------------------------------

__all__ = [
    'validate_filename',
    'safe_copy_into',
    'InsufficientResourceError',
    'GpuOutOfMemoryError',
    'OperationCancelled',
    'JOB_LOCK',
    'CancelToken',
    'available_ram_bytes',
    'total_ram_bytes',
    'require_ram',
    'estimate_ckpt_ram_mb',
    'check_disk_space',
    'estimate_output_bytes',
]
(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* safely.

    - Validates the filename via ``validate_filename``.
    - Uses ``guards.atomic_copy_into`` to ensure the copy is atomic.
    """
    # Ensure the filename is safe
    safe_name = validate_filename(src.name)
    # Construct destination path
    dest_path = dest_dir / safe_name
    # Perform atomic copy (existing helper ensures atomicity)
    atomic_copy_into(src, dest_dir)
    # Note: ``atomic_copy_into`` already writes to ``dest_dir`` with original name;
    # we rely on it to respect the validated name because ``src.name`` is safe.

def system_status_markdown(workspace_dir) -> str:
    ram_avail = available_ram_bytes()
    ram_total = total_ram_bytes()
    ram_line = (f'{ram_avail/1e9:.1f} / {ram_total/1e9:.1f} GB free'
                if ram_avail is not None and ram_total is not None else 'unknown')
    try:
        free, total = shutil.disk_usage(str(workspace_dir)).free, shutil.disk_usage(str(workspace_dir)).total
        disk_line = f'{free/1e9:.1f} / {total/1e9:.1f} GB free'
    except Exception:
        disk_line = 'unknown'
    if torch.cuda.is_available():
        try:
            free_v, total_v = torch.cuda.mem_get_info()
            gpu_line = f'{torch.cuda.get_device_name(0)} — {free_v/1e9:.1f} / {total_v/1e9:.1f} GB free VRAM'
        except Exception:
            gpu_line = 'CUDA available (details unknown)'
    else:
        gpu_line = 'no GPU detected — running on CPU'
    if JOB_LOCK.busy:
        busy_line = ('<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
                     'background:var(--m3-error-container);color:var(--m3-on-error-container);'
                     'font-weight:600;font-size:12.5px;">● a job is currently running</span>')
    else:
        busy_line = ('<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
                     'background:var(--m3-success-container);color:var(--m3-on-success-container);'
                     'font-weight:600;font-size:12.5px;">● idle</span>')
    return (f'**RAM:** {ram_line}  ·  **Disk:** {disk_line}  ·  **GPU:** {gpu_line}\n\n'
            f'**Status:** {busy_line}')

# ---------------------------------------------------------------------------
# Helper: validate filenames for uploaded files
# ---------------------------------------------------------------------------

def validate_filename(name: str) -> str:
    """Validate and sanitise a filename.

    Allows alphanumeric characters, dash, underscore, and dot. Raises
    ``ValueError`` if the name contains prohibited characters or attempts
    directory traversal.
    """
    if '..' in name or '/' in name or '\\' in name:
        raise ValueError('Filename must not contain path separators')
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError('Filename contains illegal characters')
    return name
