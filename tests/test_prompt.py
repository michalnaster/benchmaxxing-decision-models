from benchmaxx.prompt import PREFILL, PromptBuilder, display_texts, question_line
from benchmaxx.schema import Question, QuestionType


def choice() -> Question:
    return Question("Which team?", QuestionType.CHOICE, ("billing", "technical", "sales"),
                    ("payments and invoices", "", "new business"))


def test_option_spans_hold_the_option_texts(tokenizer):
    builder = PromptBuilder(tokenizer)
    for prompt in builder.readings("A ticket.", choice()):
        shown = [tokenizer.decode(prompt.ids[a:b]) for a, b in prompt.option_spans]
        assert shown == [" billing: payments and invoices", " technical", " sales: new business"]
        assert tokenizer.decode(prompt.ids).endswith(PREFILL)


def test_codes_follow_the_display_order(tokenizer):
    forward, backward = PromptBuilder(tokenizer).readings("A ticket.", choice())
    assert [chr(c) for c in forward.option_codes] == ["A", "B", "C"]
    assert [chr(c) for c in backward.option_codes] == ["C", "B", "A"]


def test_yes_no_shows_yes_first_and_asks_statements(tokenizer):
    q = Question("The ticket is urgent.", QuestionType.YES_NO, ("no", "yes"), ("", ""))
    assert display_texts(q) == ["No", "Yes"]
    assert question_line(q) == "Is the following true? The ticket is urgent."
    forward, _ = PromptBuilder(tokenizer).readings("s", q)
    assert [chr(c) for c in forward.option_codes] == ["B", "A"]  # option "yes" is shown first, as A


def test_score_marks_the_ends_and_long_states_are_cut(tokenizer):
    q = Question("How urgent?", QuestionType.SCORE, ("1", "2", "3"), ("low", "medium", "high"))
    assert display_texts(q) == ["low (lowest)", "medium", "high (highest)"]
    prompt = PromptBuilder(tokenizer, max_state_tokens=5).build("x" * 50, q, [0, 1, 2])
    assert prompt.truncated_state
