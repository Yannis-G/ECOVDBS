from abc import ABC, abstractmethod


class BaseConfig(ABC):
    
    @abstractmethod
    def to_dict(self):
        raise NotImplementedError
