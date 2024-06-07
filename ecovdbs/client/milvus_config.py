from .base_config import BaseConfig


class MilvusConfig(BaseConfig):
    """
    Configuration class for Chroma database clients.
    """

    def __init__(self, host: str = "localhost", port: int = 19530) -> None:
        """
        Initialize the ChromaConfig with default values.

        :param host: The hostname for the database server. Defaults to "localhost".
        :param port: The port number for the database server. Defaults to 8000.
        """
        self.__host: str = host
        self.__port: int = port

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        :return: The configuration as a dictionary.
        """
        return {
            "host": self.__host,
            "port": self.__port
        }