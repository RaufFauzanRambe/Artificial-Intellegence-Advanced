"""
Configuration management module for the Artificial Intelligence Advanced project.

Provides a Config dataclass that holds all project hyperparameters with support
for loading from / saving to YAML and JSON files, and overriding individual
values via environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


# ------------------------------------------------------------------
# Config dataclass
# ------------------------------------------------------------------

@dataclass
class Config:
    """Central configuration container for training, model, and data settings.

    Attributes
    ----------
    seed : int
        Global random seed for reproducibility.
    device : str
        Compute device (``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``).
    experiment_name : str
        Descriptive name used for logging directories.
    output_dir : str
        Root directory for experiment outputs.
    """

    # -- General -------------------------------------------------------
    seed: int = 42
    device: str = "auto"
    experiment_name: str = "default_experiment"
    output_dir: str = "outputs"

    # -- Model ---------------------------------------------------------
    model_name: str = "simple_nn"
    input_dim: int = 128
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    output_dim: int = 10
    dropout: float = 0.3
    activation: str = "relu"
    use_batch_norm: bool = True

    # -- Training ------------------------------------------------------
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    mixed_precision: bool = False
    gradient_clipping: float = 0.0  # 0 means disabled

    # -- Data ----------------------------------------------------------
    data_dir: str = "data"
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"
    num_workers: int = 4
    pin_memory: bool = True

    # -- Augmentation --------------------------------------------------
    augment: bool = False
    noise_std: float = 0.01

    # -- Logging -------------------------------------------------------
    log_interval: int = 10
    save_checkpoint_every: int = 5
    use_wandb: bool = False
    wandb_project: str = "ai-advanced"

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Persist the configuration to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, path: Union[str, Path]) -> None:
        """Persist the configuration to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create a Config from a dictionary, ignoring unknown keys.

        Parameters
        ----------
        d : dict

        Returns
        -------
        Config
        """
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Config
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a JSON file.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Config
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Environment variable overrides
    # ------------------------------------------------------------------

    def apply_env_overrides(self) -> "Config":
        """Override individual fields from environment variables.

        Convention: ``AI_<FIELD_NAME>`` (upper-case, matching the dataclass
        attribute name).  Lists must be provided as comma-separated values.

        Returns
        -------
        Config
            ``self`` after applying overrides (mutated in-place).

        Example
        -------
        >>> import os; os.environ["AI_EPOCHS"] = "100"
        >>> config.apply_env_overrides()
        """
        field_types: Dict[str, type] = {f.name: f.type for f in fields(self)}
        for f in fields(self):
            env_key = f"AI_{f.name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is None:
                continue

            raw_type = field_types[f.name]
            # Resolve the origin type for ``List[int]`` etc.
            origin = getattr(raw_type, "__origin__", None)

            if origin is list or (isinstance(raw_type, type) and issubclass(raw_type, list)):
                # Try parsing as a comma-separated list of ints or floats
                items = [item.strip() for item in env_val.split(",")]
                try:
                    parsed = [int(item) for item in items]
                except ValueError:
                    try:
                        parsed = [float(item) for item in items]
                    except ValueError:
                        parsed = items
                setattr(self, f.name, parsed)
            elif raw_type is bool:
                setattr(self, f.name, env_val.lower() in ("1", "true", "yes", "on"))
            elif raw_type is int:
                setattr(self, f.name, int(env_val))
            elif raw_type is float:
                setattr(self, f.name, float(env_val))
            elif raw_type is str:
                setattr(self, f.name, env_val)
            else:
                setattr(self, f.name, env_val)

            print(f"[Config] Override {f.name} = {getattr(self, f.name)} (from env)")

        return self

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of all configuration values."""
        lines = ["Config Summary", "=" * 40]
        for f in fields(self):
            val = getattr(self, f.name)
            lines.append(f"  {f.name:.<30s} {val}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Config(experiment_name={self.experiment_name!r}, "
            f"model={self.model_name!r}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, lr={self.learning_rate})"
        )


# ------------------------------------------------------------------
# Default configuration factory
# ------------------------------------------------------------------

def get_default_config() -> Config:
    """Return a fresh Config instance with all default values."""
    return Config()
