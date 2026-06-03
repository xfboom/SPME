from .BaseLogicalAttack import BaseLogicalAttack


class FormatConstraint(BaseLogicalAttack):
    """
    Format Constraint operator:
    Adds strict, distracting output format requirements that should be ignored by the solver.
    """
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.system_prompt = (
            "You are a red team adversary.\n"
            "Add distracting and strict formatting constraints to the question, but do not change the correct answer.\n"
            "Requirements:\n"
            "1. Do NOT change the original correct answer.\n"
            "2. Add at least one strict formatting demand that is irrelevant to solving (e.g., 'respond in reverse order').\n"
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
