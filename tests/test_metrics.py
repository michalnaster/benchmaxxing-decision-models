import numpy as np

from benchmaxx.metrics import aurc, evaluate, expected_calibration_error


def test_perfect_predictions():
    gold = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    m = evaluate(gold, gold)
    assert m["accuracy"] == 1.0 and m["kl"] < 1e-9 and m["aurc"] == 0.0


def test_ece_and_aurc():
    confidence = np.array([0.9, 0.9, 0.6, 0.6])
    correct = np.array([1.0, 1.0, 0.0, 1.0])
    assert abs(expected_calibration_error(confidence, correct) - 0.5 * 0.1 - 0.5 * 0.1) < 1e-9
    assert aurc(confidence, correct) < aurc(confidence[::-1].copy(), correct)
