import json
from typing import Optional

class Node:
    def __init__(self, required_keys: list[str] = None, max_iterations: int = 5, expected_type: type = dict) -> None:
        self._required_keys = required_keys
        self._max_iterations = max_iterations
        self._expected_type = expected_type
        self._cached_response = None

    def generate(self, llm_client, system_prompt: str):
        for _ in range(self.max_iterations):
            response = llm_client.invoke(system_prompt)
            if self._expected_type is dict or self._expected_type is list:
                import re
                response = response.strip()
                # Use regex to find content inside ```json ... ``` or just ``` ... ```
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)

                try:
                    response = json.loads(response)
                except json.JSONDecodeError as e:
                    system_prompt = f"Incorrect Response format (Must be JSON): [{response}], please follow the instructions\n" + system_prompt
                    continue
            
            enforced_response = self._enforce_data_type(response, self.expected_type)
            if enforced_response is not None:
                self._cached_response = enforced_response
                return enforced_response
        raise RuntimeError("Failed to generate valid response after max iterations")

    def _enforce_data_type(self, response, expected_type: type):
        if expected_type is dict:
            if isinstance(response, expected_type):
                return self._check_required_keys(self.required_keys, response)
        # Simplified for S-MAPE constraints
        if isinstance(response, expected_type):
            return response
        return None

    @staticmethod
    def _check_required_keys(required_keys: list[str], response: dict):
        if not required_keys:
            return response
        for key in required_keys:
            if key not in response:
                return None
        return response

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def expected_type(self) -> type:
        return self._expected_type

    @property
    def required_keys(self) -> list[str]:
        return self._required_keys
