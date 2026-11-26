import pytest

from benchmaxx.schema import Decision, QuestionType


def test_parses_a_choice_decision():
    d = Decision.from_dict({"case_id": 1, "state": "s", "gold": [0.7, 0.3],
                            "question": {"text": "Which?", "type": "choice",
                                         "options": [{"key": "a", "text": "first"}, {"key": "b"}]}})
    assert d.case_id == "1"
    assert d.question.option_keys == ("a", "b")
    assert d.question.option_texts == ("first", "")
    assert d.gold == (0.7, 0.3)


def test_yes_no_defaults_and_label():
    d = Decision.from_dict({"case_id": "c", "state": "s", "label": 1,
                            "question": {"text": "Is it urgent?", "type": "yes_no"}})
    assert d.question.type is QuestionType.YES_NO
    assert d.question.option_keys == ("no", "yes")
    assert d.gold == (0.0, 1.0)


def test_rejects_gold_of_the_wrong_length():
    with pytest.raises(ValueError):
        Decision.from_dict({"case_id": "c", "state": "s", "gold": [1.0],
                            "question": {"text": "q", "options": [{"key": "a"}, {"key": "b"}]}})
