import yaml
from os.path import abspath, join, dirname
from meta_classes import Singleton

__config_loc__ = join(dirname(abspath(__file__)), 'config.yaml')


class Config(metaclass=Singleton):
    """
    A configuration wrapper that parses config keys from the config.yaml file.
    """
    def __init__(self, config: dict = None):
        if config is None:
            with open(__config_loc__, 'r') as stream:
                self.conf = yaml.load(stream)

    def get_property(self, resource_name):
        conf = None
        if resource_name in self.conf:
            conf = self.conf[resource_name]
        else:
            raise KeyError("The configuration key does not exist")
        return conf
