import yaml
from os.path import abspath, join, dirname
from meta_classes import Singleton
import logging.config

__config_loc__ = join(dirname(abspath(__file__)), 'config.yaml')


class Config(metaclass=Singleton):
    """
    A configuration wrapper that parses config keys from the config.yaml file.
    """

    def __init__(self, config: dict = None, logger_level=None):
        if config is None:
            with open(__config_loc__, 'r') as stream:
                self.conf = yaml.load(stream)

        if logger_level is None:
            logger_level = logging.INFO

        if 'logger' in self.conf:
            logger_conf = self.conf['logger']
            logging.config.dictConfig(logger_conf)
        else:
            logging.basicConfig(level=logger_level)

    def get_property(self, resource_name):
        """
        A function that returns a config value or dict
        :param resource_name: the config name
        :return: a string or dictionary of properties
        """
        conf = None
        if resource_name in self.conf:
            conf = self.conf[resource_name]
        else:
            raise KeyError("The configuration key does not exist")
        return conf
