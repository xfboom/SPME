# red_team/BaseLogicalAttack.py (Abstract class adapter)
from abc import ABC, abstractmethod

class BaseLogicalAttack(ABC):
    """
    Base abstract class for logical attack operators
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    @abstractmethod
    def apply_mutation(self, original_question: str) -> str:
        """
        Mutate the original question. Subclasses must implement this method.
        """
        pass
