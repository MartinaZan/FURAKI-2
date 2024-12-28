from abc import ABC, abstractmethod

class Splitter(ABC):
    """Base class for splitter objects.
    A splitter object detects a drift and performs a split"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def check(self):
        pass