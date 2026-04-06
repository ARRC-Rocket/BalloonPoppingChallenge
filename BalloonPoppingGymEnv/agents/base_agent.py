from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, given_parameters):
        """
        Args:
            given_parameters (dict): A dictionary of rocket, environment, and balloon parameters provided to the agent.
        """
        super().__init__()
        self.given_parameters = given_parameters

    @abstractmethod
    def get_action(self, observation):
        raise NotImplementedError("Must be implemented by child class")
