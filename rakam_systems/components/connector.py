from abc import ABC, abstractmethod

class Connector(ABC):
    """
    An abstract base class for components that can be extended for various functionalities.
    """
    def __init__(self, **kwargs) -> None:
        pass
    
    @abstractmethod
    def call_main(self, **kwargs) -> dict:
        """
        Main method for executing the component's functionality.
        """
        pass
    
    @abstractmethod
    def test(self, **kwargs) -> bool:
        """
        Method for testing the component.
        """
        pass


