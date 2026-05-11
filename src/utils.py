"""
Utility functions and helpers for the Artificial Intelligence Advanced project.

Includes seed management, timing, metric tracking, JSON I/O, device detection,
and model parameter counting.
"""

from __future__ import annotations

import json
import os
import random
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python stdlib, NumPy, and (optionally) PyTorch / CUDA.

    Ensures deterministic behaviour for experiments.

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic CuDNN (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("PyTorch seed set to %d (CuDNN deterministic mode enabled).", seed)
    except ImportError:
        logger.debug("PyTorch not available — seed set for Python and NumPy only.")

    logger.info("Global random seed set to %d.", seed)


# ------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------

@contextmanager
def Timer(name: str = "operation", enabled: bool = True) -> Generator[None, None, None]:
    """Context manager that measures elapsed wall-clock time.

    Parameters
    ----------
    name : str
        Label used in the log message.
    enabled : bool
        If *False*, timing is skipped entirely (zero overhead).

    Example
    -------
    >>> with Timer("training"):
    ...     train_model()
    [training] elapsed: 12.34 s
    """
    if not enabled:
        yield
        return

    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("[%s] elapsed: %.2f s", name, elapsed)
    print(f"[{name}] elapsed: {elapsed:.2f} s")


# ------------------------------------------------------------------
# Metric tracking
# ------------------------------------------------------------------

class AverageMeter:
    """Running average tracker for training / evaluation metrics.

    Keeps a sliding window of recent values and computes their mean.

    Parameters
    ----------
    name : str
        Human-readable label (e.g., ``"loss"``, ``"accuracy"``).
    window_size : int or None
        Maximum number of recent values to average over.  If *None*, all
        values since the last ``reset`` are used.

    Example
    -------
    >>> meter = AverageMeter("loss")
    >>> meter.update(0.5); meter.update(0.3); meter.update(0.2)
    >>> meter.avg
    0.333...
    """

    def __init__(self, name: str = "", window_size: Optional[int] = None) -> None:
        self.name = name
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        """Clear all recorded values."""
        self._values: List[float] = []
        self._sum = 0.0
        self._count = 0
        self.avg = 0.0
        self.val = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """Record a new observation.

        Parameters
        ----------
        value : float
            The metric value for this step.
        n : int
            Number of items represented by *value* (useful for batch-aware averaging).
        """
        self.val = value
        self._sum += value * n
        self._count += n
        self._values.append(value)

        # Trim to window size if configured
        if self.window_size is not None and len(self._values) > self.window_size:
            dropped = self._values.pop(0)
            self._sum -= dropped

        window = self._values[-self.window_size:] if self.window_size else self._values
        self.avg = sum(window) / len(window) if window else 0.0

    @property
    def global_avg(self) -> float:
        """Average over *all* values since the last reset."""
        return self._sum / self._count if self._count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.4f}, count={self._count})"


# ------------------------------------------------------------------
# JSON I/O helpers
# ------------------------------------------------------------------

def save_json(data: Union[Dict, List], filepath: Union[str, Path], indent: int = 2) -> None:
    """Serialize *data* to a JSON file.

    Parameters
    ----------
    data : dict or list
        Data to write.
    filepath : str or Path
        Destination path.
    indent : int
        Pretty-print indentation level.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=_json_serializer)
    logger.info("Saved JSON to %s", filepath)


def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
    """Load JSON data from a file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    dict or list
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded JSON from %s", filepath)
    return data


def _json_serializer(obj: Any) -> Any:
    """Fallback serializer for types not natively handled by ``json``."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


# ------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------

def get_device(device_str: Optional[str] = None) -> str:
    """Return the best available compute device.

    Parameters
    ----------
    device_str : str, optional
        Explicit device string (e.g., ``"cpu"``, ``"cuda"``, ``"cuda:0"``).
        If *None*, auto-detects CUDA availability.

    Returns
    -------
    str
        Device identifier.
    """
    if device_str is not None:
        return device_str

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


# ------------------------------------------------------------------
# Model parameter counting
# ------------------------------------------------------------------

def count_parameters(model: Any, trainable_only: bool = True) -> Dict[str, int]:
    """Count the parameters of a PyTorch (or Keras-like) model.

    Parameters
    ----------
    model : Any
        A model with a ``parameters()`` method (PyTorch) or ``count_params``
        attribute (Keras).
    trainable_only : bool
        If *True*, only trainable parameters are counted.

    Returns
    -------
    dict
        Keys ``"total"``, ``"trainable"``, ``"non_trainable"``.
    """
    result: Dict[str, int] = {"total": 0, "trainable": 0, "non_trainable": 0}

    try:
        import torch
        for param in model.parameters():
            n = param.numel()
            result["total"] += n
            if param.requires_grad:
                result["trainable"] += n
            else:
                result["non_trainable"] += n
    except (ImportError, AttributeError):
        # Fallback for non-PyTorch models
        try:
            result["total"] = model.count_params()  # type: ignore[attr-defined]
            result["trainable"] = result["total"]
        except AttributeError:
            logger.warning("Unable to count parameters for model of type %s.", type(model).__name__)

    if trainable_only:
        return {"total": result["trainable"]}
    return result
