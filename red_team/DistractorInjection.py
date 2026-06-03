from .BaseLogicalAttack import BaseLogicalAttack

class DistractorInjection(BaseLogicalAttack):
    """
    Distractor Injection operator:
    Calls LLM to inject irrelevant distractor numbers or background information into the original question,
    to test the robustness of the LLM's reasoning and attention mechanism under strong interference.
    """
    def __init__(self, llm_client):
        super().__init__(llm_client)
        # Fixed system prompt for this red team operator
        self.system_prompt = (
            "You are a highly stealthy red team logic trap generation expert.\n"
            "Your task is to randomly and naturally weave some [irrelevant distractor numbers or background information] into the original logic question provided by the user.\n"
            "Requirements:\n"
            "1. The injected information must appear to be highly relevant to the core logic of the question, but practically it must not affect the final derivation at all.\n"
            "2. The original correct answer of the question must not be changed.\n"
            "3. Add at least three distractors: two numeric and one contextual.\n"
            "4. Include at least one misleading unit or conversion cue that should be ignored.\n"
            "5. Include one irrelevant constraint or exception clause that appears important but is not.\n"
            "6. Prefer subtle traps such as extra conditions, reversed causal phrasing, or decoy quantities that mirror real ones.\n"
            "7. Only return the mutated question text. It is forbidden to add any greetings, explanations, or other formatting prefix symbols (no Markdown code block needed)."
        )

    def apply_mutation(self, original_question: str) -> str:
        """
        Generate a trap question with noise information based on the original question.
        """
        # We send the simplest instruction with clear defense boundaries to the LLM
        prompt_to_llm = f"{self.system_prompt}\n\n[Original Logic Question]\n{original_question}\n\n[Please output the question after injecting traps]:"
        
        # Call the core underlying invoke API
        adversarial_question = self.llm_client.invoke(prompt_to_llm)
        
        # Return the mutated content directly
        return adversarial_question.strip()
