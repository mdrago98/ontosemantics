import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from meta_classes import Singleton

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "bioengine" / "config" / "default.yaml"
_RESOURCE_ROOT = _PACKAGE_ROOT / "resources"
_CONFIG_ENV_VAR = "BIOENGINE_CONFIG_PATH"


class Config(metaclass=Singleton):
    """Configuration facade that loads defaults packaged with bioengine."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger_level: Optional[int] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            config_path = os.getenv(_CONFIG_ENV_VAR)
        self.package_root = _PACKAGE_ROOT
        self._default_resource_root = _RESOURCE_ROOT
        self._config_path = Path(config_path).expanduser() if config_path else None
        data, base_dir = self._load_config_dict(config, self._config_path)
        self.conf = self._normalise(data, base_dir)
        self.conf.setdefault("resource_dir", str(self._default_resource_root))

        if logger_level is None:
            self.logger_level = logging.INFO
        else:
            self.logger_level = logger_level

        self.logger_conf = self.conf.get("logger")

    def _load_config_dict(
        self,
        config: Optional[Dict[str, Any]],
        config_path: Optional[Path],
    ) -> Tuple[Dict[str, Any], Path]:
        if config is not None:
            base_dir = config_path.parent if config_path else self.package_root
            return config, base_dir

        if config_path is not None:
            cfg_path = config_path
        else:
            cfg_path = _DEFAULT_CONFIG_PATH

        if not cfg_path.exists():
            raise FileNotFoundError(f"Unable to locate configuration file at {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        return data, cfg_path.parent

    def _normalise(self, data: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}

        substitutions = {
            "${PACKAGE_ROOT}": str(self.package_root),
            "${RESOURCE_DIR}": str(self._default_resource_root),
        }

        def resolve(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: resolve(val) for key, val in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, str):
                expanded = os.path.expandvars(os.path.expanduser(value))
                for token, replacement in substitutions.items():
                    expanded = expanded.replace(token, replacement)
                if base_dir and expanded.startswith("./"):
                    expanded = str((base_dir / expanded).resolve())
                return expanded
            return value

        return resolve(data)

    def get_property(self, resource_name: str) -> Any:
        if resource_name not in self.conf:
            raise KeyError("The configuration key does not exist")
        return self.conf[resource_name]
