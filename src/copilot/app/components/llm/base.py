from abc import ABC, abstractmethod
from typing import List

class LLM(ABC):
    """
    Abstract base class for a Large Language Model (LLM).

    This class should be subclassed and its abstract methods implemented.

    Methods
    -------
    generate(messages: List[dict]) -> str
        Generate an answer based on input messages using an LLM.
    stream()
        Stream an answer based on input messages using an LLM.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[dict],
    ) -> str:
        """
        Generate an answer based on input messages using an LLM.

        Parameters
        ----------
        messages : List[dict]
            A list of messages. Each message is a dictionary containing the necessary information for the LLM to generate an answer.

        Returns
        -------
        str
            The generated answer.
        """

    @abstractmethod
    def stream(self):
        """
        Stream an answer based on input messages using an LLM.

        This method should be implemented to handle streaming of answers.
        """
