from .BaseLogicalAttack import BaseLogicalAttack


class LogicInversion(BaseLogicalAttack):
    """
    Logic Inversion operator:
    Introduces inverted or opposing conditions as decoys while preserving the original answer.
    """
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.system_prompt = (
            "You are a red team logic adversary.\n"
            "Your task is to inject decoy statements that invert or negate key conditions, but those decoys must be irrelevant to the correct answer.\n"
            "Requirements:\n"
            "1. Do NOT change the original correct answer.\n"
            "2. Add at least one inverted condition (e.g., 'greater than' vs 'less than') that is clearly a decoy.\n"
            "3. Keep the question coherent and natural.\n"
            "4. Return only the mutated question text, no extra formatting or explanations."
        )

    def apply_mutation(self, original_question: str) -> str:
        prompt_to_llm = (
            f"{self.system_prompt}\n\n"
            f"[Original Question]\n{original_question}\n\n"
            "[Output the mutated question only]:"
        )
        adversarial_question = self.llm_client.invoke(prompt_to_llm)
        return adversarial_question.strip()
