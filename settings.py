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
            self.logger_level = logging.INFO
        else:
            self.logger_level = logger_level

        if 'logger' in self.conf:
            self.logger_conf = self.conf['logger']
        else:
            self.logger_conf = None

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
