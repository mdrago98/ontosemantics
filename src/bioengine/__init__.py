
import logging
from logging.config import dictConfig

from . import nlp_processor, scripts, dochandlers, cypher_engine, utils

from src.settings import Config
if Config().logger_conf is not None:
    dictConfig(Config().logger_conf)
else:
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
