from BaseConfig import BaseConfig


class ChromaConfig(BaseConfig):
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.__host: str = host
        self.__port: int = port

    def to_dict(self):
        return {
            "host": self.__host,
            "port": self.__port
        }
