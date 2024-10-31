from abc import ABC, abstractmethod

from rakam_systems.system_manager import SystemManager

class Component(ABC):
    """
    An abstract base class for components that can be extended for various functionalities.
    """
    def __init__(self, system_manager: SystemManager, **kwargs) -> None:
        self.system_manager = system_manager
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

