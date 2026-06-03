import json
from core.LLMClient import LLMClient

class LLMJudge:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def describe(self) -> str:
        return (
            "Judge logic: compare question, ground truth, and model response to assess correctness."
        )

    def evaluate(self, question: str, target_response: str, correct_answer: str, verbose: bool = True) -> bool:
        """
        Use an LLM judge to decide whether the target response is correct.
        """
        prompt = f"""You are an objective and rigorous judge. Based on the question and ground truth, decide whether the target response is correct.

1) Question:
{question}

2) Ground Truth:
{correct_answer}

3) Target Response:
{target_response}

Analyze the target response:
- Is the reasoning coherent?
- Was it misled by distractors?
- Is the final conclusion semantically consistent with the ground truth?

Return ONLY a JSON object in this format (no Markdown code fences):
{{
    "reasoning": "Your concise reasoning (20-50 words)",
    "is_correct": true or false
}}
"""
        try:
            if verbose:
                print(
                    "      [Judge Input] "
                    f"question_len={len(question)}, "
                    f"answer={correct_answer}, "
                    f"response_len={len(target_response)}"
                )
            response_text = self.llm_client.invoke(prompt)
            # Strip Markdown code fences to extract JSON
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1]
            
            result = json.loads(json_str.strip())
            
            # Print judge reasoning for debugging
            if verbose:
                print(f"      [Judge Note] {result.get('reasoning', 'No reasoning')}")
                print(f"      [Judge Result] is_correct={result.get('is_correct', False)}")
            
            return result.get("is_correct", False)
            
        except Exception as e:
            print(f"      [Judge Error] JSON parse failed: {e}. Raw output: {response_text}")
            # Fallback: simple substring match
            return correct_answer in target_response
