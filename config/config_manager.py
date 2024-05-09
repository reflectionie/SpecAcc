import yaml

class Config:
    _instance = None

    # def __new__(cls, config_path=None):
    #     if cls._instance is None:
    #         cls._instance = super(Config, cls).__new__(cls)
    #         if config_path is not None:
    #             try:
    #                 with open(config_path, 'r') as file:
    #                     cls._instance.config = yaml.safe_load(file)
    #             except Exception as e:
    #                 print(f"Error reading the config file: {e}")
    #                 cls._instance.config = {}
    #     return cls._instance
    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            if config_path is not None:
                try:
                    with open(config_path, 'r') as file:
                        config = yaml.safe_load(file)
                        for key, value in config.items():
                            if isinstance(value, str) and value.lower() == 'none':
                                config[key] = None
                        cls._instance.config = config
                except Exception as e:
                    print(f"Error reading the config file: {e}")
                    cls._instance.config = {}
        return cls._instance


    @staticmethod
    def get_config(config_path=None):
        return Config(config_path)._instance.config
