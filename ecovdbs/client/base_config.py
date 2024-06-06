from abc import ABC, abstractmethod


class BaseConfig(ABC):
    """
    Abstract base class defining the configuration interface for database clients.
    """

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        :return: The configuration as a dictionary.
        """
        raise NotImplementedError
