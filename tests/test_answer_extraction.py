from scripts.aaai_utils import extract_answer, is_correct_output


def test_extracts_strict_answer_tag():
    assert extract_answer("<answer>(D)</answer>") == "(D)"
    assert is_correct_output("<answer>D</answer>", "(D)")


def test_ignores_placeholder_answer_tag_and_uses_final_answer_label():
    output = (
        "Output exactly one line: <answer>FINAL_ANSWER</answer>\n"
        "</think>\n\n"
        "ANSWER: False"
    )
    assert extract_answer(output) == "False"
    assert is_correct_output(output, "False")


def test_extracts_boxed_choice():
    output = "</think>\n\nThus, the correct answer is:\n\n\\boxed{E}"
    assert extract_answer(output) == "(E)"
    assert is_correct_output(output, "(E)")


def test_extracts_markdown_answer_label_with_next_line():
    output = "</think>\n\n**Answer:**\n(A) The mangoes are the cheapest"
    assert extract_answer(output) == "(A)"
    assert is_correct_output(output, "(A)")


def test_extracts_answer_label_choice_letter():
    assert extract_answer("</think>\n\nANSWER: D") == "(D)"
    assert is_correct_output("</think>\n\nANSWER: D", "(D)")
