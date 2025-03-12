from abc import ABC, abstractmethod


class BaseModule(ABC):

    @abstractmethod
    def init(self):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
