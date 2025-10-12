import logging
from logging.config import dictConfig

from settings import Config

_config = Config()
if _config.logger_conf is not None:
    dictConfig(_config.logger_conf)
else:
    logging.basicConfig(level=_config.logger_level)

logger = logging.getLogger(__name__)

__all__ = ["Config", "logger"]
