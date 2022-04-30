from configparser import ConfigParser


def get_config():
    config = ConfigParser()
    config.read("app_config.ini")
    return config
