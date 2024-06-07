from .base_config import BaseConfig, BaseIndexConfig


class MilvusConfig(BaseConfig):
    """
    Configuration class for Milvus database clients.
    """

    def __init__(self, connection_uri: str = "http://localhost:19530") -> None:
        """
        Initialize the MilvusConfig with default values.

        :param connection_uri: The connection URI for the Milvus server. Defaults to "http://localhost:19530".
        """
        self.__uri = connection_uri

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        :return: The configuration as a dictionary.
        """
        return {
            "uri": self.__uri
        }
