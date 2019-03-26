
import logging
from logging.config import dictConfig

from settings import Config
if Config().logger_conf is not None:
    dictConfig(Config().logger_conf)
else:
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
